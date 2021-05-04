import os
import shutil
import warnings
import subprocess
import torch
import atomium
import tempfile
import numpy as np
import pandas as pd
import concurrent.futures
from Bio.SeqUtils import seq1

from zipfile import ZipFile
from rossmann_toolbox.utils import sharpen_preds, corr_seq, custom_warning
from rossmann_toolbox.utils import SeqVec
from rossmann_toolbox.utils.encoders import SeqVecMemEncoder
from rossmann_toolbox.utils.generators import SeqChunker
from rossmann_toolbox.models import SeqCoreEvaluator, SeqCoreDetector
from conditional import conditional
from captum.attr import IntegratedGradients

from rossmann_toolbox.utils import MyFoldX, fix_TER, solveX
from rossmann_toolbox.utils import separate_beta_helix
from rossmann_toolbox.utils.tools import run_command, extract_core_dssp
from rossmann_toolbox.utils import Deepligand3D
from rossmann_toolbox.utils.graph_feat_prep import feats_from_stuct_file
from csb.bio.io.hhpred import HHOutputParser
warnings.showwarning = custom_warning


class RossmannToolbox:
	struct_utils = None
	def __init__(self, n_cpu = -1, use_gpu=True, foldx_loc=None, hhsearch_loc=None, dssp_loc=None):
		"""
		Rossmann Toolbox - A framework for predicting and engineering the cofactor specificity of Rossmann-fold proteins
		:param n_cpu: Number of CPU cores to use in the CPU-dependent calculations. n_cpu=-1 will use all available cores
		:param use_gpu: Use GPU to speed up predictions
		:param foldx_loc: optional absolute path to foldx binary file required for structure-based predictions
		:param hhsearch_loc: Location of hhsearch binary (v3.*) required for hhsearch-enabled Rossmann core detection
		:param dssp_loc: Location of the DSSP binary required for structure-based predictions
		"""
		self.label_dict = {'FAD': 0, 'NAD': 1, 'NADP': 2, 'SAM': 3}
		self.rev_label_dict = {value: key for key, value in self.label_dict.items()}
		self.n_classes = 4

		self.n_cpu = n_cpu if n_cpu != -1 else os.cpu_count()
		# Handle GPU config
		self.use_gpu = False
		self.device = torch.device('cpu')
		if use_gpu:
			if torch.cuda.is_available():
				self.device = torch.device('cuda')
				self.use_gpu = True
			else:
				warnings.warn('No GPU detected, falling back to the CPU version!')

		self._path = os.path.dirname(os.path.abspath(__file__))
		self._seqvec = self._setup_seqvec()
		self._weights_prefix = f'{self._path}/weights/'
		self._foldx_loc = foldx_loc
		if self._foldx_loc is not None:
			if not self._check_foldx():
				raise RuntimeError(
					'Foldx v4 binary not detected in the specified location: \'{}\''.format(self._foldx_loc))
		else:
			warnings.warn('FoldX binary location was not provided. The structure-based prediction functionality will be disabled.')

		self._hhsearch_loc = hhsearch_loc
		if self._hhsearch_loc is not None:
			if not self._check_hhsearch():
				raise RuntimeError(
					'HHsearch v3 binary not detected in the specified location: \'{}\''.format(self._hhsearch_loc))
		else:
			warnings.warn(
				"HHpred path was not provided. The HHsearch-based prediction of Rossmann cores won't be available")

		self._dssp_loc = dssp_loc
		if self._dssp_loc is not None:
			if not self._check_dssp():
				raise RuntimeError(
					'DSSP binary not detected in the specified location: \'{}\''.format(self._dssp_loc))
		else:
			warnings.warn('DSSP binary location was not provided. The structure-based prediction functionality will be disabled.')

		if self._foldx_loc is not None and self._dssp_loc is not None:
			self.dl3d = self._setup_dl3d()

	def _check_hhsearch(self):
		try:
			output = subprocess.check_output(self._hhsearch_loc, universal_newlines=True)
		except subprocess.CalledProcessError as e:
			output = e.output
		except (FileNotFoundError, PermissionError):
			return False
		return output.split('\n')[0].startswith('HHsearch 3')

	def _check_foldx(self):
		try:
			output = subprocess.check_output(self._foldx_loc, universal_newlines=True)
		except subprocess.CalledProcessError as e:
			output = e.output
		except (FileNotFoundError, PermissionError):
			return False
		if 'foldX time has expired' in output.split('\n')[-8]:
			raise RuntimeError('FoldX5 license expired, renew the binary and restart RossmannToolbox!')
			return False
		return 'FoldX 4' in output.split('\n')[2]

	def _check_dssp(self):
		try:
			output = subprocess.check_output(self._dssp_loc, universal_newlines=True, stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			output = e.output
		except (FileNotFoundError, PermissionError):
			return False
		return output.split('\n')[0].startswith('mkdssp 3')

	def _run_hhsearch(self, sequence, min_prob=0.5):
		temp = tempfile.NamedTemporaryFile(mode='w+t')
		temp.writelines(">seq\n{}\n".format(sequence))
		temp.seek(0)
		fn = temp.name
		cmd = f'{self._hhsearch_loc} -i {fn} -d {self._path}/utils/hhdb/core -n 1'
		result = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		temp.close()
		if result == 0:
			out_fn = f'{fn}.hhr'
			parser = HHOutputParser()
			hits = {i: (hit.qstart, hit.qend, hit.probability) for i, hit in
					enumerate(parser.parse_file(out_fn)) if hit.probability >= min_prob}
			os.remove(out_fn)

			# Choose highest prob hit from overlapping hits
			hits_nr = {}
			for (beg, end, prob) in hits.values():
				found_overlap = False
				res_set = {i for i in range(beg, end + 1)}
				for key, hit in hits_nr.items():
					hit_set = {i for i in range(hit[0], hit[1] + 1)}
					if len(hit_set & res_set) >= 0:
						if prob > hit[2]:
							hits_nr[key] = (beg, end, prob)
						found_overlap = True
						break
				if not found_overlap:
					hits_nr[len(hits_nr)] = (beg, end, prob)
			probs = [0]*len(sequence)
			for (beg, end, prob) in hits_nr.values():
				for i in range(beg, end+1):
					probs[i] = prob
			return hits_nr, probs
		return {}, ()

	def _process_input(self, data, full_length=False):
		"""
		Validate and process the input eventually filtering/correcting problematic sequences
		:param data: Input data - a dictionary with ids and corresponding sequences as keys and values
		:param full_length: Denotes whether full-length sequences are validated (True/False)
		:return: pd.DataFrame for internal processing
		"""
		if not isinstance(data, dict):
			raise ValueError('Input data must be a dictionary with ids and sequences as keys and values!')
		valid_seqs = [isinstance(seq, str) for seq in data.values()]
		if not all(valid_seqs):
			raise ValueError('Input data must be a dictionary with ids and sequences as keys and values!')
		if len(data) == 0:
			raise ValueError('Empty dictionary was passed as an input!')
		non_std_seqs = [sequence != corr_seq(sequence) for sequence in data.values()]
		if not full_length:
			too_short_seqs = [len(sequence) < 20 for sequence in data.values()]
			too_long_seqs = [len(sequence) > 65 for sequence in data.values()]
			if all(too_short_seqs):
				raise ValueError('Input sequence(s) are below 20aa length!')

			if all(too_short_seqs):
				raise ValueError('Input sequence(s) are above 65aa length!')

		# Convert input dict to pd.DataFrame for easy handling
		data = pd.DataFrame.from_dict(data, orient='index', columns=['sequence'])

		if any(non_std_seqs):
			data['sequence'] = data['sequence'].apply(lambda x: corr_seq(x))
			warnings.warn('Non-standard residues detected in input data were corrected to X token.', UserWarning)
		if not full_length:
			if any(too_short_seqs):
				data = data[data['sequence'].str.len() >= 20]
				warnings.warn('Filtered out input sequences shorter than 20 residues.', UserWarning)

			if any(too_long_seqs):
				data = data[data['sequence'].str.len() <= 65]
				warnings.warn('Filtered out input sequences longer than 65 residues.', UserWarning)

		return data

	def _setup_seqvec(self):
		"""
		Load SeqVec model to either GPU or CPU, if weights are not cached - download them from the mirror.
		:return: instance of SeqVec embedder
		"""
		seqvec_dir = f'{self._path}/weights/seqvec'
		seqvec_conf_fn = f'{seqvec_dir}/uniref50_v2/options.json'
		seqvec_weights_fn = f'{seqvec_dir}/uniref50_v2/weights.hdf5'
		if not (os.path.isfile(seqvec_conf_fn) and os.path.isfile(seqvec_weights_fn)):
			print('SeqVec weights are not available, downloading from the remote source (this\'ll happen only once)...')
			torch.hub.download_url_to_file('https://rostlab.org/~deepppi/seqvec.zip',
										   f'{self._path}/weights/seqvec.zip')
			archive = ZipFile(f'{self._path}/weights/seqvec.zip')
			archive.extract('uniref50_v2/options.json', seqvec_dir)
			archive.extract('uniref50_v2/weights.hdf5', seqvec_dir)
		if self.use_gpu:
			return SeqVec(model_dir=f'{seqvec_dir}/uniref50_v2', cuda_device=0, tokens_per_batch=8000)
		else:
			return SeqVec(model_dir=f'{seqvec_dir}/uniref50_v2', cuda_device=-1, tokens_per_batch=8000)
	
	def _setup_dl3d(self):
		"""
		Load DL3D model to either GPU or CPU
		:return: structural graph predictor instance
		"""
		weights_dir = f'{self._path}/weights'
		device_type = 'cpu'
		return Deepligand3D(weights_dir, device_type)

	@staticmethod
	def _filter_cores(data, detected_cores):
		"""
		Parses output of `seq_detect_cores` to dictionary with core sequences
		:param data: Input data - a dictionary with ids and corresponding full sequences as keys and values
		:param detected_cores: Input data - a dictionary with ids and corresponding sequences as keys and values
		:return: dictionary with extracted Rossmann sequences
		"""
	
		# Check for undetected cores
		detected_ids = {key for key, value in detected_cores.items() if len(value) > 0}
		passed_ids = set(data.keys())
		if len(set(detected_ids)) != len(set(passed_ids)):
			missing_ids = set(passed_ids) - set(detected_ids)
			warnings.warn('Rossmann cores were not detected for ids: {}'.format(', '.join(missing_ids)))

		# Check for multiple cores in one sequence
		multiple_hits_ids = {str(key) for key, value in detected_cores.items() if len(value[0]) > 1}
		if len(multiple_hits_ids) > 0:
			warnings.warn(
				'Found multiple Rossmann cores for ids: \'{}\'. Passing first hit for further predictions'.format(
					', '.join(multiple_hits_ids)))
		cores_filtered = {key: data[key][value[0][0][0]:value[0][0][1]] for key, value in detected_cores.items() if len(value[0]) > 0}
		data = cores_filtered
		return data

	def _setup_seq_core_detector(self):
		return SeqCoreDetector().to(self.device)

	def _run_seq_core_detector(self, data):
		embeddings = self._seqvec.encode(data, to_file=False)
		seqvec_enc = SeqVecMemEncoder(embeddings, pad_length=500)
		gen = SeqChunker(data, batch_size=64, W_size=500, shuffle=False,
						 data_encoders=[seqvec_enc], data_cols=['sequence'])
		model = self._setup_seq_core_detector()
		model.load_state_dict(torch.load(f'{self._weights_prefix}/coredetector.pt'))
		model = model.eval()
		preds = []

		# Predict core locations
		with torch.no_grad():
			for batch in gen:
				batch = torch.tensor(batch[0].astype(dtype=np.float32), device=self.device)
				batch_preds = model(batch)
				preds.append(batch_preds.cpu().detach().numpy())

		# Post-process predictions
		preds = np.vstack(preds)
		preds_depadded = {key: np.concatenate([preds[ix][ind[0]:ind[1]] for ix, ind in zip(*value)]) for key, value
						  in gen.indices.items()}
		return {key: (sharpen_preds(value), value) for key, value in preds_depadded.items()}

	def _setup_seq_core_evaluator(self):
		return SeqCoreEvaluator().to(self.device)

	def seq_detect_cores(self, data, mode='dl'):
		"""
		Detects Rossmann beta-alpha-beta cores in full-length protein sequences.
		:param data: Input data - a dictionary with ids and corresponding sequences as keys and values
		:param mode: Mode of Rossmann core detection - either 'hhsearch' or 'dl'.
		:return: dictionary with ids and detected core locations as keys and values and per residue probabilities.
		"""
		data = self._process_input(data, full_length=True)

		if mode == 'hhsearch':
			executor_ = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cpu)
			with executor_ as executor:
				futures = {executor.submit(self._run_hhsearch, sequence): key for key, sequence in
						   data['sequence'].to_dict().items()}
				cores = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}
			return cores
		elif mode == 'dl':
			cores = self._run_seq_core_detector(data)
			return cores

	def seq_evaluate_cores(self, data, importance = False):
		"""
		Predicts the cofactor specitificty of the Rossmann core sequences
		:param data: Input data - a dictionary with ids and corresponding sequences as keys and values
		:param importance: Return additional per-residue importances, i.e. contributions to the final
		specificity predictions
		:return: Dictionary with the sequence ids and per-sequence predictions of the cofactor specificties.
		If importance is True each dictionary value will contain additional per-residue importances.
		"""

		data = self._process_input(data)
		# Initialize importances and preds per fold values which latter will be averaged
		preds_ens = []
		attrs_ens = []
		# Encode with SeqVec
		embeddings = self._seqvec.encode(data, to_file=False)

		# Setup generator that'll be evaluated
		seqvec_enc = SeqVecMemEncoder(embeddings, pad_length=65)
		gen = SeqChunker(data, batch_size=64, W_size=65, shuffle=False,
						 data_encoders=[seqvec_enc], data_cols=['sequence'])

		# Predict with each of N predictors, depad predictions and average out for final output
		model = self._setup_seq_core_evaluator()
		model = model.eval()
		for i in range(0, 5):
			model.load_state_dict(torch.load(f'{self._weights_prefix}/{i}.pt'))
			preds = []
			attrs = []
			with conditional(not importance, torch.no_grad()):
				# Raw predictions
				for batch in gen:
					batch = torch.tensor(batch[0].transpose(0, 2, 1).astype(dtype=np.float32), device=self.device)
					batch_preds = model(batch)
					preds.append(batch_preds.cpu().detach().numpy())
					if importance:
						ig = IntegratedGradients(model)
						baseline = torch.full_like(batch, 0, device=self.device)
						batch_attrs = np.asarray(
							[ig.attribute(batch, baseline, i).sum(axis=1).clip(min=0).cpu().detach().numpy() for i in
							 range(0, self.n_classes)])
						attrs.append(batch_attrs.transpose(1, 0, 2))

			preds_ens.append(np.vstack(preds))
			if importance:
				attrs_ens.append(np.vstack(attrs))

		# Average predictions between all predictors
		preds_ens = np.asarray(preds_ens)
		attrs_ens = np.asarray(attrs_ens)
		avgs = preds_ens.mean(axis=0)
		stds = preds_ens.std(axis=0)
		results = {key: {f'{self.rev_label_dict[i]}{suf}': val[i] for i in range(0, self.n_classes) for suf, val in
						 zip(['', '_std'], [avg, std])} for key, avg, std in zip(data.index, avgs, stds)}
		if importance:
			avgs = attrs_ens.mean(axis=0)
			stds = attrs_ens.std(axis=0)
			attrs = {key: {f'{self.rev_label_dict[j]}':
							   ((np.concatenate([avgs[ix, j, ind[0]:ind[1]] for ix, ind in zip(*value)]),
								 np.concatenate([stds[ix, j, ind[0]:ind[1]] for ix, ind in zip(*value)]))
							   ) for j in range(0, self.n_classes)}
					 for key, value in gen.indices.items()}
			return results, attrs
		return results

	def _prepare_struct_files(self, pdb_chains):
		"""
		creates all files needed for futher calculations
		:param pdb_chains: list of chains
		"""

		#checks if raw struct file exists
		for chain in pdb_chains:
			if not self.struct_utils.is_structure_file_cached(chain):
				self.struct_utils.download_pdb_chain(chain)
		
		#checks if foldx struct file exists 
		for chain in pdb_chains:
			if not self.struct_utils.is_foldx_file_cached(chain):
				self.struct_utils.repair_pdb_chain(chain)
		
		#checks if foldx feat files exists 
		for chain in pdb_chains:
			if not self.struct_utils.is_foldx_feat_file_cached(chain):
				self.struct_utils.calc_struct_feats(chain)        
	
	def _prepare_struct_feats(self, pdb_chains, mode, core_detect_mode, core_list):
		"""
		extract all features needed to run DL3D
		"""
		if self.struct_utils is None:
			raise RuntimeError(' structural utilities are not initialized properly')
		
		for chain_pos, chain in enumerate(pdb_chains):
			if chain is None:
				raise ChainNotFound('chain %s not found in: %s ' %(chain, path))
			_, chain_id = chain.split('_')  
							
			# raw structure as provided by the user
			path_pdb_file = os.path.join(self.struct_utils.path, chain) + '.pdb'
			chain_struct = atomium.open(path_pdb_file)
			chain_struct = chain_struct.model.chain(chain_id)
			
			# "...if seq1(res.name)!='X'" removed because we want to include non-canonical residues too
			pdb_res = [res for res in chain_struct.residues() if res.full_name not in ['water']]
				
			# pdb ids for each residue
			pdbids = [res.id.split('.')[1] for res in pdb_res]
			pdbseq = "".join([seq1(res.name) for res in pdb_res]) 

			# detect cores
			if mode=='seq':
				data = {chain:pdbseq}
				detected_cores = self.seq_detect_cores(data, mode=core_detect_mode)				
				filtred_cores = self._filter_cores(data, detected_cores)
			# use core sequence provided by the user
			else:
				filtred_cores = {chain:core_list[chain_pos]}

			# file minimised in foldX
			path_pdb_rep_file = os.path.join(self.struct_utils.path, chain) + '_Repair.pdb'
			
			frame_list = list()
			for pdb_chain, core_seq in filtred_cores.items():
				core_pos = solveX.solveX_rev(core_seq, pdbseq)[0]
				
				if core_pos == -1:
					raise ValueError(f'could not map the core ({core_seq}) onto the chain sequence:\n\n{pdbseq}')
					
				pdb_list = pdbids[core_pos:core_pos+len(core_seq)]
				assert len(pdb_list) == len(core_seq)
								
				raw_ss = extract_core_dssp(path_pdb_rep_file, pdb_list, core_seq, dssp_loc=self._dssp_loc)
				extended_ss = separate_beta_helix(raw_ss)

				frame_list.append({
					'pdb_chain' : pdb_chain,
					'seq' : core_seq,
					'pdb_list' : pdb_list,
					'fname' : path_pdb_rep_file,
					'secondary' : extended_ss
				})
				
		frame = pd.DataFrame(frame_list)
		
		# extract and process foldX and other structural features 
		foldx_info, distances_dict, edge_dict = feats_from_stuct_file(frame, self._path)    
		  
		return {'dataframe' : frame,
				'contact_maps' : distances_dict,
				'edge_feats'  : edge_dict,
				'foldx_info' : foldx_info}
								
	def struct_evaluate_cores(self, path, chain_list, mode, core_detect_mode, core_list):
								
		self.struct_utils = StructPrep(path, self._foldx_loc)
		self._prepare_struct_files(chain_list)
		data = self._prepare_struct_feats(chain_list, mode, core_detect_mode, core_list)                                    
		return data                     
								
	def predict(self, data, mode = 'core', core_detect_mode = 'dl', importance = False):
		"""
		Evaluate cofactor specificity of full-length or Rossmann-core sequences.
		:param data: Input data - a dictionary with ids and corresponding sequences as keys and values
		:param mode: Prediction mode - either 'seq' for full-sequence input or 'core' for Rossmann-core sequences
		:param core_detect_mode: Mode of Rossmann core detection. Either 'hhsearch' or 'dl'. Works only in the 'seq' mode.
		:param importance: Return additional per-residue importances, i.e. contributions to the final
		specificity predictions
		:return: Dictionary with the sequence ids and per-sequence predictions of the cofactor specificties.
		If importance is True will return additional dictionary with per-residue importances for each entry.
		If mode is 'seq' output will additionally contain the 'sequence' keys for each input id indicating the detected
		Rossmann core sequence.
		"""
		if mode not in ['seq', 'core']:
			raise ValueError('Prediction mode must be either \'seq\' or \'core\'!')
		if core_detect_mode not in ['hhsearch', 'dl']:
			raise ValueError('Core dection mode must be either \'hhsearch\' or \'dl\'!')

		# Full length sequence input
		if mode == 'seq':
			detected_cores = self.seq_detect_cores(data, mode=core_detect_mode)
			data = self._filter_cores(data, detected_cores)

		if importance:
			predictions, attrs = self.seq_evaluate_cores(data, importance=importance)
		else:
			predictions = self.seq_evaluate_cores(data, importance=importance)
		if mode == 'seq':
			for key in predictions.keys():
				predictions[key]['sequence'] = data[key]
	
		if importance:
			return predictions, attrs
		else:
			return predictions

	def predict_structure(self, path_pdb='', chain_list=None, mode = 'core', core_detect_mode = 'dl', core_list=None, importance = False):
		"""
		structure-based prediction
		:param path_pdb: path to directory with pdb structures & foldx data
		:param chain_list: list of chains used in predictions, if chain is not available it will be downloaded
        :return Dictionary with the sequence ids and per-sequence predictions of the cofactor specificties
        If importance is True
		"""
	
		if not os.path.isabs(path_pdb):
			raise ValueError('The variable \'path_pdb\' has to be the absolute path!')	
	
		if mode not in ['seq', 'core']:
			raise ValueError('Prediction mode must be either \'seq\' or \'core\'!')
		if core_detect_mode not in ['hhsearch', 'dl']:
			raise ValueError('Core dection mode must be either \'hhsearch\' or \'dl\'!')
			
		if mode == 'core':
			if core_list==None:
				raise ValueError('For the \'core\' mode please provide a list of core sequences \'core_list\'')	 
			else:
				assert len(chain_list) == len(core_list), 'the number of chains must equal to the number of cores'
	
		if self._foldx_loc is None or self._dssp_loc is None:
			raise RuntimeError(
				'Locations of binaries (DSSP and FoldX) were not specified. Re-run `RossmannToolbox` with `foldx_loc`, and `dssp_loc`')
	
		if not os.path.isdir(path_pdb):
			raise NotADirectoryError(f'given path_pdb: {path_pdb} is not a directory')
	
		if chain_list is None:
			chain_list = os.listdir(path_pdb)
			#uses only those files with format XXXX_Y.pdb where XXXX is protein id and Y is a chain
			chain_list = [f.replace('.pdb', '') for f in chain_list if f.endswith('.pdb')]
			chain_list = [f for f in chain_list if len(f) == 6]
		
		self.feats3d = self.struct_evaluate_cores(path_pdb, chain_list, mode, core_detect_mode, core_list)
		results = self.dl3d.predict(**self.feats3d).to_dict(orient='records')

		if importance:
			results_imp = self.dl3d.generate_embeddings(**self.feats3d)
			return results, results_imp
		else:
			return results
        	
class StructPrep:
	foldx_suffix = '_Repair.pdb'
	foldx_feat_suffix = '_Repair_PN.fxout'
	rotabase  = 'rotabase.txt'
	def __init__(self, path, path_foldx_bin):
		"""
		structure preparation flow for node and edge features extraction
		"""
		self.path = path
		self.path_foldx_bin = path_foldx_bin
		self.path_foldx = os.path.dirname(path_foldx_bin)
		if not os.path.isfile(self.path_foldx_bin):
			raise FileNotFoundError('foldx binary not found in', self.path_foldx_bin)
		self._read_cache()
		if 'rotabase.txt' not in self.files:
			shutil.copyfile(os.path.join(self.path_foldx, self.rotabase), os.path.join(self.path, self.rotabase))

	def download_pdb_chain(self, pdb_chain):
		"""
		downloads certain protein chain via atomium library
		:params: pdb_chain - wothout .pdb extension
		"""
		path_dest = os.path.join(self.path, pdb_chain)
		struc_id, chain = pdb_chain.split('_')
		temp = atomium.fetch(struc_id.upper())
		temp.model.chain(chain.upper()).save(path_dest + '.pdb')

	def repair_pdb_chain(self, pdb_chain):
		"""
		repairs pdb file with foldx `RepairPDB` command
		"""
		print(f'Preparing {pdb_chain} for FoldX. This may take a while but the result will be cached.')
		pdb_chain = pdb_chain + '.pdb' if not pdb_chain.endswith('.pdb') else pdb_chain
		work_dir = os.getcwd()
		#change working directory to `PATH_CACHED_STRUCTURES` without that
		#foldx cant find structure error `No pdbs for the run found at: "./" Foldx will end`
		os.chdir(self.path)
		cmd = f'{self.path_foldx_bin} --command=RepairPDB --pdb={pdb_chain}'
		out = run_command(cmd)
		fix_TER(pdb_chain)
		os.chdir(work_dir)
	
	def calc_struct_feats(self, pdb_chain):
		# calculate structural features using foldX 
		fx = MyFoldX(self.path, self.path, self.path_foldx_bin)
		fx._MyFoldX__calc_foldx_features(self.path, pdb_chain + self.foldx_suffix)
	
	def _read_cache(self):
		"""
		read content of `path` variable
		"""
		if not os.path.isdir(self.path):
			raise NotADirectoryError(f'structure dir :{self.path}')
	
		self.files = os.listdir(self.path)
		self.files_struct = [f for f in self.files if f.find(self.foldx_suffix) == -1]
		self.files_foldx = [f for f in self.files if f.find(self.foldx_suffix) != -1]
		self.files_foldx_feats = [f for f in self.files if f.find(self.foldx_feat_suffix) != -1]
	  
	def is_structure_file_cached(self, file_pdb):

		condition = False
		file_pdb = file_pdb + '.pdb' if not file_pdb.endswith('.pdb') else file_pdb
		path_file_full = os.path.join(self.path, file_pdb)
		if file_pdb in self.files_struct:
			if os.path.getsize(path_file_full) > 0:
				condition = True
		return condition

	def is_foldx_feat_file_cached(self, file_pdb):
	
		res1 = f'SD_Optimized_{file_pdb.replace(".pdb", "")}_Repair.fxout'
		res2 = f'InteractingResidues_Hbonds_Optimized_{file_pdb.replace(".pdb", "")}_Repair_PN.fxout'
		
		if res2 not in self.files_foldx_feats: return False
		
		# check whether the two files are empty
		condition = False
		for r in [res1, res2]:
			path_file_full = os.path.join(self.path, r)
			if os.path.getsize(path_file_full) > 0:
				condition = True
		return condition
		
	def is_foldx_file_cached(self, file_pdb):

		condition = False
		file_pdb = file_pdb + self.foldx_suffix if not file_pdb.endswith(self.foldx_suffix) else file_pdb
		path_file_full = os.path.join(self.path, file_pdb)
		if file_pdb in self.files_foldx:
			if os.path.getsize(path_file_full) > 0:
				condition = True
		return condition
    

	

	
	
	
