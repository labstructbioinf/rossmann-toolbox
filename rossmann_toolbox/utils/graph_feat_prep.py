import sys, os
import gzip, pickle
import torch, dgl

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix

from packaging import version

import Bio.PDB as bp
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from .bio_params import CM_THRESHOLD
from .foldx_parser import foldx_details_parser, align_edges, align_nodes


def fix_rama(d):
	"""
	phi, psi angles returned by foldx may contain erros. let's fix them!
	"""
	while True:
		nanpos = np.where(d == -9999.)[0]
		if len(nanpos)>0:
			try:
				d[nanpos] = d[nanpos+1]
			except IndexError:
				d[nanpos] = d[nanpos-1]
			
		if not any(d == -9999.): break
	return d

def generate_Cb(residue):
	'''
	creates phantom Cb atom
	'''
	n=residue['N'].get_vector()
	c=residue['C'].get_vector()
	ca=residue['CA'].get_vector()
	n=n-ca
	c=c-ca
	rot=bp.rotaxis(-np.pi*120.0/180.0, c)
	cb_at_origin=n.left_multiply(rot)
	return np.array(list(cb_at_origin+ca ))

def side_chains_angles(C_a, C_b):

	'''
	AB x AC = ||AB|| ||AC|| cosx
	'''

	num_residues = max(C_a.shape)
	angles = np.empty((num_residues, num_residues), dtype=np.float32)
	for i in range(num_residues):
		cai = C_a[i]
		cbi = C_b[i]
		for j in range(num_residues):
			caj = C_a[j]
			cbj = C_b[j]
			C_a_diff = cai - caj
			cbj_fx = cbj - C_a_diff 
			AB = cai - cbi
			AC = cai - cbj_fx
			cosx = np.dot(AB,AC)/(np.dot(AB, AB) * np.dot(AC, AC))
			angle = np.arccos(cosx)
			angles[i,j] = angle
			angles[j,i] = angle
			if i < j:
				break
	return angles

def calculate_adjecency(fname, pdb_chain, pdb_list, seq, get_angle=True, include_nones=True, **kw_args):
	'''
	params:
		fname (str) path to .pdb structure file
		chain (str) chain name
		pdb_list (list) list of residues indices
		seq (str) chain residues sequence - used to match sizes
	returns:
		distance_matrix (np.ndarray) 
	'''
	#check informations
	assert pdb_chain is not None
	assert os.path.isfile(fname), f'no such file {fname}'

	with open(fname, 'rt') as f:
		structure = bp.PDBParser().get_structure(pdb_chain,f)
	assert len(structure)==1
	model = structure[0]

	# Generate dict of residue objects
	resid2res = dict([(''.join([str(j) for j in res.full_id[-1][1:]]).strip(), res) for res in model.get_residues()])
	coords_a = []
	coords_b = []
	for pdb_idx in pdb_list:
		# Residue present in sequence but *not* in structure
		if pdb_idx is None and include_nones == False:
			continue
		elif pdb_idx is None:
			coords_a.append(np.array([np.NaN, np.NaN, np.NaN]))
			coords_b.append(np.array([np.NaN, np.NaN, np.NaN]))
		else:
			res = resid2res[pdb_idx]
			coords_a.append(res.child_dict['CA'].coord)
		
			if res.resname == 'GLY':
				cb = generate_Cb(res)
			else:
				try:
					cb = res.child_dict['CB'].coord
				except KeyError:
					print(f'CB atom missing for res {res.resname} in {pdb_chain}; dispatching monkeys to address this fatal issue')
					cb = generate_Cb(res)
			coords_b.append(cb)
		
	if include_nones == True:        
		assert len(coords_a)==len(coords_b)==len(seq)
	
	xyz_alpha = np.array(coords_a, dtype=np.float32)
	xyz_beta = np.array(coords_b, dtype=np.float32)
	#alpha_dist = prairwaise_vec(xyz_alpha)
	#beta_dist = prairwaise_vec(xyz_beta)

	# To sieje bledami z powodu nan'ow (swoja droga eleganckie rozwiazanie!)
	#parallel_side_chains = (alpha_dist < beta_dist)*1 
	distance = distance_matrix(xyz_alpha, xyz_alpha)
	if None in pdb_list and include_nones == True:
		shape = distance.shape[0]
		off_diag_left = np.arange(1, shape, 1, dtype=int)
		off_diag_right = np.arange(0, shape-1,1,dtype=int)
		diag = np.arange(0, shape, 1, dtype=int)
		distance[off_diag_left,off_diag_right] = 5
		distance[off_diag_right,off_diag_left] = 5
	if get_angle:     
		angle_dist = side_chains_angles(xyz_alpha, xyz_beta)
		return distance, angle_dist
	else:
		return distance

def get_efeats(dst, ang):
	'''
	extracts from Ca-Ca and (Ca-Cb) - (Ca-Cb) matrices
	and produces basic node features 
	'''
	if version.parse(dgl.__version__) < version.parse('0.5.2'):
		g = dgl.DGLGraph(dst < CM_THRESHOLD)
	else:
		nx_graph = nx.Graph(dst < CM_THRESHOLD)
		g = dgl.from_networkx(nx_graph)

	edge_features = np.empty((g.number_of_edges(),4))
	ev1, ev2 = g.edges()
	for i,(e1, e2) in enumerate(zip(ev1, ev2)):
		e = np.abs(e1 - e2) != 1
		e_seq = not e
		feats = [1/(dst[e1,e2]+1e-2), 
				 ang[e1,e2], \
				 e,
				e_seq]
		edge_features[i, :] = feats
	return edge_features

def align_sequence_with_structrue(pdb_list, sequence, secondary):

	'''
	removes Nones from sequence data
	'''

	assert len(pdb_list) == len(sequence) == len(secondary)    

	pdb_list = np.asarray(pdb_list)
	sequence = np.asarray(list(sequence))
	secondary = np.asarray(list(secondary))
	nonan_pos = np.where(np.asarray(pdb_list) != None)[0]

	new_sequence = "".join(list(sequence[nonan_pos]))
	new_secondary = "".join(list(secondary[nonan_pos]))
	new_pdb_list = list(pdb_list[nonan_pos])
	
	return new_pdb_list, new_sequence, new_secondary


def transform_edges_to_graph_shape(edge_feats, dst):
	'''
	reshape foldx feature matrix size (n,n,num_feats) with array
	(num_edges, num_features) with cutoff `params.CM_THRESHOLD
	'''

	assert edge_feats.shape[0] == dst.shape[0]

	if version.parse(dgl.__version__) < version.parse('0.5.2'):
		g = dgl.DGLGraph(dst < CM_THRESHOLD)
	else:
		nx_graph = nx.Graph(dst < CM_THRESHOLD)
		g = dgl.from_networkx(nx_graph)
	
	num_nodes, _, num_features = edge_feats.shape
	edge_features = np.empty((g.number_of_edges(), num_features))

	ev1, ev2 = g.edges()

	for i,(e1, e2) in enumerate(zip(ev1, ev2)):
		feats = [feat for feat in edge_feats[e1, e2, :].ravel()]
		edge_features[i, :] = feats
	
	return edge_features

def validate_input(df, columns=['seq', 'secondary', 'pdb_list']):
	'''
	returns True if columns size dont match
	'''
	errors = 0
	for _, row in df.iterrows():

		if len(row.seq) != len(row.secondary) != len(row.pdb_list):
			errors += 1
	return errors



def load_gpickle(file):
    '''
    returns content of gziped (optional requires .gz extension) pickle file
    '''
    
    assert isinstance(file, str)
    assert os.path.isfile(file), f'no such file {file}'
    
    if file.endswith('.gz'):
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    return data

def feats_from_stuct_file(data, ref_path):
	'''
	:params: dataframe with columns (fname, pdb_chain, pdb_list, seq)
	'''
	
	# === calculate distances
	distances_dict = dict()
	for idx, row in data.iterrows():
		distances_dict[idx] = calculate_adjecency(**row, get_angle=True)

	# === collect and process foldX data
	foldx_info = dict()
	
	# nodes features must be scaled
	foldx_scaler = f'{ref_path}/weights/struct_ensemble/scaler.pkl.gz'
	scaler = load_gpickle(foldx_scaler)
	
	parser = foldx_details_parser()
	for idx, row in data.iterrows():
		base_name = os.path.dirname(row.fname)
		edge_res, node_res = parser.get_data(base_name, row.pdb_chain)

		# Select and process nodes' features
		foldx_nodes = align_nodes(node_res, row.pdb_list, row.seq, row.pdb_chain)
		
		# fix phi, psi angles
		foldx_nodes[:,0] = fix_rama(foldx_nodes[:,0])
		foldx_nodes[:,1] = fix_rama(foldx_nodes[:,1])
		assert -9999. not in foldx_nodes, 'unable to fix missing phi, psi angles'
		
		# transform nodes features
		foldx_nodes = scaler.transform(foldx_nodes)
		
		
		# Build a graph and fill edge data
		foldx_edges = align_edges(edge_res, row.pdb_list, row.seq, row.pdb_chain)	
		foldx_edges_reshaped = transform_edges_to_graph_shape(foldx_edges, distances_dict[idx][0])
		foldx_info[idx] = {'node_data' : foldx_nodes, 'edge_data' : foldx_edges_reshaped}

	# === process edge features
	edge_dict = dict()
	for seq, vals in distances_dict.items():
		edge_dict[seq] = get_efeats(*vals)

	return foldx_info, distances_dict, edge_dict


