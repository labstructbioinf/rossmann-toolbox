import sys
import warnings

import pandas as pd
import numpy as np

from Bio.SeqUtils import seq1
from .struct_model_hparams import hparams

cols="""Pdb
three_letter
chain
pdb_seq_num
omega
phi
psi
sec_struct
total
backHbond
sideHbond
energy_VdW
electro
energy_SolvP
energy_SolvH
energy_vdwclash
entrop_sc
entrop_mc
sloop_entropy
mloop_entropy
cis_bond
energy_torsion
backbone_vdwclash
energy_dipole
water
disulfide
energy_kon
partcov
energyIonisation
Hetero Backbone HBond
entr_complex
Hetero Sidechain Hbond
Sidechain Accessibility
Mainchain Accessibility
Sidechain Contact Ratio
Mainchain Contact Ratio
ab_index"""
cols = cols.split('\n')


radius = {
	'A': 88.6,
	'R': 173.4,
	'G': 60.1,
	'S': 89.0,
	'C': 108.5,
	'D': 111.1,
	'P': 112.7,
	'N': 114.1,
	'T': 116.1,
	'E': 138.4,
	'Q': 143.8,
	'H': 153.2,
	'M': 162.9,
	'I': 166.7,
	'L': 166.7,
	'K': 168.6,
	'F': 189.9,
	'Y': 193.6,
	'W': 227.8,
	'V': 140.0
}
phobos = {
	'I': 4.5,
	'V': 4.2,
	'L': 3.8,
	'F': 2.8,
	'C': 2.5,
	'M': 1.9,
	'A': 1.8,
	'G': -0.4,
	'T': -0.7,
	'S': -0.8,
	'W': -0.9,
	'Y': -1.3,
	'P': 1.6,
	'H': -3.2,
	'D': -3.5,
	'E': -3.5,
	'N': -3.5,
	'Q': -3.5,
	'K': -3.9,
	'R': -4.5
}


DTYPE_EDGES = np.bool_

class foldx_details_parser():
	def __init__(self, results_dir="", debug=False):
		self.results_dir = results_dir
		self.debug = debug

	def __parse(self, f_name):

		def get_res(s):
			reject = ['MG', 'NA', 'ZN', 'CA', 'KA', 'CU', 'MN', 'FE', 'CO', 'KB']

			if len(s)<4: return None		
			aa, chainb, pos = s[:3], s[3], s[4:]
		
			if not pos.replace('-', '').isnumeric() or not chainb.isalpha():
				p=True
				for r in reject:
					if s.startswith(r):
						p=False
						break
				if p: print('wrong aa, excluded: ', s)
			
				return None
			else:
				return (aa, chainb, pos)

		if self.debug: print('parsing', f_name)

		f = open(f_name, 'rt')
		res = {}

		PARSE=False
		l = f.readline()
		while l:
			l = l.strip('\n')
	
			if l.startswith('-'):
				header = pre_l
				if self.debug: print('---', header)
				assert not header in res
				res[header] = {}
				PARSE=True
			elif PARSE:
				if l=='': 
					PARSE=False
				else:
					if not l.startswith("Residue"):
						residues = l.split('\t')
						from_res = get_res(residues[0])
						if from_res!=None:						
							to_res = [get_res(i) for i in residues[1:]]
							to_res = [i for i in to_res if i!=None]
							if not from_res in res[header]:
									res[header][from_res] = to_res
							else:
								warnings.warn(f"Duplicated residue {from_res}")
				
			pre_l = l            
			l = f.readline()
   
		assert len(res)==4

		return res


	def get_data(self, path, pdb_chain, my_data_dir="", model_idx=""):

		data_types = ['Hbonds', 'Volumetric', 'Electro', 'VdWClashes']
        
		#if my_data_dir=="":
        #    mid = pdb_chain[1:3]
		#	 data_dir = f'{self.results_dir}/{mid}/{pdb_chain}/'
		#else:
        #    data_dir = my_data_dir
        
        #data_dir = os.path.join(self.results_dir, pdb_chain)
		edge_res = {}
		for dt in data_types:
            
			f_name = f'{path}/InteractingResidues_{dt}_Optimized_{pdb_chain}_Repair{model_idx}_PN.fxout'
			edge_res[dt] = self.__parse(f_name)
	
		node_res = pd.read_csv(f'{path}/SD_Optimized_{pdb_chain}_Repair{model_idx}.fxout', 
							  sep='\t', names=cols, dtype={'pdb_seq_num':str, })

		tmp=len(node_res)
		node_res.drop_duplicates(subset='pdb_seq_num', inplace=True)
		if tmp!=len(node_res):
			warnings.warn(f"{tmp-len(node_res)} duplicated residues removed from node_res")
		
		node_res.drop(node_res[node_res.pdb_seq_num.isna()].index, inplace=True)

		assert node_res.pdb_seq_num.is_unique
		node_res.set_index('pdb_seq_num', inplace=True, drop=False)
	
		return edge_res, node_res
	
def align_edges(foldx_edge_interactions, pdb_list, seq, pdb_chain):
	assert foldx_edge_interactions

	pdb, chain = pdb_chain.split("_")

	residues_mapping = {pdb_idx : matrix_idx for matrix_idx, pdb_idx in enumerate(pdb_list)}
	num_residues = len(pdb_list)
	pdb_list_set = set(pdb_list)

	frame_list = []

	#interaction force
	for interaction in foldx_edge_interactions.values():

		#interaction type
		for int_type in interaction.values():
			frame = np.zeros((num_residues,num_residues),dtype=DTYPE_EDGES)
			#residue - residue binary labels
			for res_pdb, res_interact_list_pdb in int_type.items():
				if not res_pdb[2] in pdb_list: continue
				assert res_pdb[1] == chain, (res_pdb, chain)
			
				# convert from pdb numbering to array index
				res = residues_mapping[res_pdb[2]]
				tmp_aa = seq1(res_pdb[0].title())
				if tmp_aa!="X": assert seq[res] == tmp_aa, (seq[res], res_pdb, tmp_aa)
							
				res_interact_list_pdb = [r[2] for r in res_interact_list_pdb]
				res_interact_list = [residues_mapping[r] for r in set(res_interact_list_pdb) & pdb_list_set]

				for ri in res_interact_list:
						frame[res, ri] = 1

			frame_list.append(frame[:,:,np.newaxis])
		frame_array = np.concatenate(frame_list, axis=2)
	return frame_array

def align_nodes(foldx_nodes_features, pdb_list, seq, pdb_chain):
	# selected features
	#feats = ['phi', 'psi', 'Sidechain Accessibility', 'Mainchain Accessibility']

	feats = ['phi', 'psi', 'total', 'backHbond', 'sideHbond', 'energy_VdW', 'electro', 'energy_SolvP',
			'energy_SolvH', 'energy_vdwclash', 'entrop_sc', 'entrop_mc', 'cis_bond',
			'energy_torsion', 'backbone_vdwclash', 'energy_dipole', 'Sidechain Contact Ratio',
			'Mainchain Contact Ratio']
		
	assert len(feats)+2 == hparams['in_dim_n']

	# check sequence
	tmp_seq = "".join([seq1(aa.title()) for aa in foldx_nodes_features.loc[foldx_nodes_features.pdb_seq_num.isin(pdb_list)].three_letter.tolist()])
	assert len(seq) == len(tmp_seq)
	for i, j in zip(list(seq), list(tmp_seq)):
		if j!="X": assert i==j
	
	# add foldx features
	nodes_feats = foldx_nodes_features.loc[foldx_nodes_features.pdb_seq_num.isin(pdb_list), feats].copy()

	# add more features
	nodes_feats['phobos'] = [phobos[aa] for aa in list(seq)]
	nodes_feats['radius'] = [radius[aa] for aa in list(seq)]

	return nodes_feats.values.astype(np.float32)

