#http://foldxsuite.crg.eu/command/PositionScan
#https://github.com/JinyuanSun/Codes_for_FoldX/blob/master/multiple_threads_foldx_positionscan.py
#https://github.com/SBRG/ssbio/blob/master/ssbio/protein/structure/utils/foldx.py


"""
- highly stabilising (ΔΔG < −1.84 kcal/mol);
- stabilising (−1.84 kcal/mol ≤ ΔΔG < −0.92 kcal/mol);
- slightly stabilising (−0.92 kcal/mol ≤ ΔΔG < −0.46 kcal/mol);
- neutral (−0.46 kcal/mol < ΔΔG ≤ +0.46 kcal/mol);
- slightly destabilising (+0.46 kcal/mol < ΔΔG ≤ +0.92 kcal/mol);
- destabilising (+0.92 kcal/mol < ΔΔG ≤ +1.84 kcal/mol);
- highly destabilising (ΔΔG > +1.84 kcal/mol).
"""


import os, glob
from Bio.Data.IUPACData import protein_letters_3to1
import pandas as pd
from .tools import run_command, extract_core, extract_core_dssp


def fix_TER(pdb_file):

	f = open(pdb_file, 'rt')
	for line in f:
		pass
	last = line.strip()
	f.close()
	
	if last!='TER':
		print(f'TER added to {pdb_file}')
		f = open(pdb_file, 'a')
		f.write('TER\n')
		f.close()


class MyFoldX():
	def __init__(self, templates_path, models_path, foldx_path):
		assert os.path.isabs(models_path)
		self.models_path = models_path
		assert os.path.isabs(templates_path)
		self.templates_path = templates_path
		assert os.path.isfile(foldx_path)
		self.foldx_path = foldx_path
		
	def model_mutants(self, mutations_df):
		assert not any(mutations_df.mutations.str.contains('ins')) and  \
			   not any(mutations_df.mutations.str.contains('del')), 'FoldX cannot handle indels!'
			   
		work_dir = os.getcwd()
			   
		for ref_pdb_chain, cores in mutations_df.groupby('ref_pdb_chain'):
		
			print(f'{len(cores)} mutation set(s) for template {ref_pdb_chain}')
		
			pdb, chain = ref_pdb_chain.split("_")
			pdb_chain = ref_pdb_chain	
			
			wt_pdb_list = cores.iloc[0].wt_pdb_list
			wt_seq = cores.iloc[0].wt_seq 
			assert all([all(c.wt_pdb_list == wt_pdb_list) for _, c in cores.iterrows()])		   

			template_pdb_file = f'{self.templates_path}{pdb}_{chain}_Repair.pdb'
			if not os.path.isfile(template_pdb_file):
				os.chdir(self.templates_path)
				# Get full structure
				cmd = f'gunzip /home/db/localpdb/pdb_chain/{pdb[1:3]}/{pdb}_{chain}.pdb.gz -c > {pdb_chain}.pdb'
				tools.run_command(cmd)
				
				# Minimize in FoldX
				cmd = f'{self.foldx_path} --command=RepairPDB --pdb={pdb_chain}.pdb'
				tools.run_command(cmd)
				os.chdir(work_dir)
				
				# Structures from FoldX lack the ending TER
				fix_TER(template_pdb_file)
			else:
				print('Template read from cache...')
							
			# extract core and core DSSP			
			wt_core_pdb = tools.extract_core(template_pdb_file, chain, wt_pdb_list, wt_seq)			
			wt_core_pdb = wt_core_pdb.replace(self.templates_path, '')
			wt_core_dssp = tools.extract_core_dssp(template_pdb_file, wt_pdb_list, wt_seq)
				
			mutations_df.loc[cores.index, 'wt_ss_dssp'] = wt_core_dssp
			mutations_df.loc[cores.index, 'wt_core_pdb'] = wt_core_pdb
				
			# Model the mutations
			foldx_result_path = f'{self.models_path}/{pdb_chain}/Dif_{pdb}_{chain}_Repair.fxout'
			submodels_path = f'{self.models_path}/{pdb_chain}'

			if not os.path.isfile(foldx_result_path):			
				os.chdir(self.models_path)
				if not os.path.isdir(pdb_chain): os.mkdir(pdb_chain)
				os.chdir(submodels_path)
				# Link the wt full structure
				tools.run_command(f'ln -s {self.templates_path}/{pdb_chain}_Repair.pdb .')
				
				# Write foldX config file
				f=open('config.cfg', 'wt')
				f.write(f'command=BuildModel\n')
				f.write(f'pdb={pdb_chain}_Repair.pdb\n')
				f.write(f'mutant-file=individual_list.txt\n')
				f.close()	
				
				# Define mutations
				f=open('individual_list.txt', 'wt')
				for _, core in cores.iterrows():
					if core.mutations=='': continue
					f.write(','.join([f'{m[0]}{chain}{m[1:-1]}{m[-1]}' for m in core.mutations.split('.')]) + ';\n')
				f.close()
				tools.run_command(f'{self.foldx_path} -f config.cfg')
				os.chdir(work_dir)
			else:
				print('Models read from cache...')
				
			# Assign models' cores and dssp
			print ('Extracting mutated cores...')
			cores_idx = [i for i in cores.index if cores.loc[i].mutations!='']
			df_foldx = pd.read_csv(foldx_result_path, skiprows=8, sep='\t')
			assert len(cores_idx) == len(df_foldx)
			
			for core_idx, foldx_idx in zip(cores_idx, df_foldx.index):
				core = cores.loc[core_idx]
				foldx = df_foldx.loc[foldx_idx]
				
				full_mpdb_file = f'{self.models_path}/{pdb_chain}/{foldx.Pdb}'
				fix_TER(full_mpdb_file)
				
				chain = core.ref_pdb_chain.split("_")[-1]
								
				# Extract core
				mut_core_pdb = tools.extract_core(full_mpdb_file, chain, core.wt_pdb_list, core.mut_seq)
				mut_core_pdb = mut_core_pdb.split('/')[-1]
				mut_core_pdb = f'{pdb}_{chain}/{foldx.Pdb}'.replace('.pdb', '.core.pdb')
				mutations_df.at[core.name, 'mut_core_pdb'] = mut_core_pdb
				
				# Extract DSSP
				mut_core_dssp = tools.extract_core_dssp(full_mpdb_file, core.wt_pdb_list, core.mut_seq)
				mutations_df.at[core.name, 'mut_ss_dssp'] = mut_core_dssp
				
				# Add ddG
				mutations_df.at[core.name, 'ddG'] = foldx['total energy']
				
			# Set features for WT (not mutated sequences):
			wt_idx = [i for i in cores.index if cores.loc[i].mutations=='']
			mutations_df.at[wt_idx, 'ddG'] = 0
			mutations_df.at[wt_idx, 'mut_core_pdb'] = mutations_df.loc[wt_idx].wt_core_pdb
			mutations_df.at[wt_idx, 'mut_ss_dssp'] = mutations_df.loc[wt_idx].wt_ss_dssp
			
			
			
	def __calc_foldx_features(self, location, repaired_pdb):
		work_dir = os.getcwd()
		
		res1 = os.path.join(location, f'SD_Optimized_{repaired_pdb.replace(".pdb", "")}.fxout')
		res2 = os.path.join(location, f'InteractingResidues_Hbonds_Optimized_{repaired_pdb.replace(".pdb", "")}_PN.fxout')
				
		if not os.path.isfile(res1) or not os.path.isfile(res2):	
			print(f'Calculating FoldX features for {repaired_pdb}...')	
			os.chdir(location)
			cmd = f'{self.foldx_path} --command=Optimize --pdb={repaired_pdb}'
			run_command(cmd)
			cmd = f'{self.foldx_path} --command=SequenceDetail --pdb=Optimized_{repaired_pdb}'
			run_command(cmd)
			cmd = f'{self.foldx_path} --command=PrintNetworks --pdb=Optimized_{repaired_pdb}'
			run_command(cmd)		
			os.chdir(work_dir)
		else:
			print(f'FoldX features for {repaired_pdb} read from cache...')
					
	def add_foldx_features(self, mutations_df):
		for idx, mut in mutations_df.iterrows():
			self.__calc_foldx_features(self.templates_path, mut.wt_core_pdb.replace('.core.pdb', '.pdb'))
			self.__calc_foldx_features(f'{self.models_path}/{mut.ref_pdb_chain}', mut.mut_core_pdb.split('/')[-1].replace('.core.pdb', '.pdb'))
		
	def add_foldx_features_modeller(self, mutations_df):
		
		def qucik_repair(pdb_file, path):
			work_dir = os.getcwd()
			
			if not os.path.isfile(os.path.join(path, pdb_file)):
				os.chdir(path)

				# Minimize in FoldX
				input_pdb = pdb_file.replace("_Repair", "")
				cmd = f'{self.foldx_path} --command=RepairPDB --pdb={input_pdb}'
				tools.run_command(cmd)
				
				if pdb_file.find('best')!=-1:
					os.rename(pdb_file.replace('.best_Repair', '_Repair'), pdb_file) 
				# Structures from FoldX lack the ending TER
				fix_TER(pdb_file)
				
				os.chdir(work_dir)
			else:
				print('FoldX repaired model read from cache...')
	
		for idx, mut in mutations_df.iterrows():
			pdb, chain = mut.ref_pdb_chain.split("_")
			
			# Repair full-length Modeller template in FoldX
			template_pdb_file = f'{pdb}_{chain}_Repair.pdb'
			qucik_repair(template_pdb_file, self.templates_path)
			full_tpdb_file = os.path.join(self.templates_path, template_pdb_file)
				
			# extract wt core and wt core DSSP			
			wt_core_pdb = tools.extract_core(full_tpdb_file, chain, mut.wt_pdb_list, mut.wt_seq)				
			wt_core_pdb = wt_core_pdb.replace(self.templates_path, '')
			wt_core_dssp = tools.extract_core_dssp(full_tpdb_file, mut.wt_pdb_list, mut.wt_seq)
			
			mutations_df.loc[idx, 'wt_ss_dssp'] = wt_core_dssp
			mutations_df.loc[idx, 'wt_core_pdb'] = wt_core_pdb
			
			# Extract FoldX features for WT
			self.__calc_foldx_features(self.templates_path, template_pdb_file)
	
			# Repair full-length Modeller model in FoldX
			model_pdb_file = f'{mut.mut_core_pdb.split("/")[1].replace(".core.pdb", "")}_Repair.pdb'
			models_path = self.models_path + mut.mut_core_pdb.split('/')[0] + '/'	
			qucik_repair(model_pdb_file, models_path)
			full_mpdb_file = os.path.join(models_path, model_pdb_file)	
		
			# extract mut core and mut core DSSP
			mut_core_pdb = tools.extract_core(full_mpdb_file, chain, mut.mut_pdb_list, mut.mut_seq)
			mut_core_pdb = mut_core_pdb.replace(self.models_path, '')
			mut_core_dssp = tools.extract_core_dssp(full_mpdb_file, mut.mut_pdb_list, mut.mut_seq)									
		
			mutations_df.at[idx, 'mut_core_pdb'] = mut_core_pdb
			mutations_df.at[idx, 'mut_ss_dssp'] = mut_core_dssp
			
			# Extract FoldX features for mut
			self.__calc_foldx_features(models_path, model_pdb_file)
			
	
			
			
		
	