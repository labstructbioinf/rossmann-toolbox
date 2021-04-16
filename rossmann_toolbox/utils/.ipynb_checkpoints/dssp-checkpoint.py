from Bio.PDB.DSSP import _make_dssp_dict
import pandas as pd
import gzip

def get_dssp_seq(fn):
	"""
	Extracts sequence from DSSP output file.
	:param fn: input filename
	:return: sequence in PDB structure
	# TODO: Connectivity info??
	"""
	f = open(fn, 'r')
	out_dict, keys = _make_dssp_dict(f)
	seq = [(out_dict[key][0]) for key in keys]
	f.close()
	return ''.join(seq)

def parse_dssp_output(dssp_fn, use_gzip=False):

	if use_gzip:
		f = gzip.open(dssp_fn, 'r')
	else:
		f = open(dssp_fn, 'r')
	
	lines = [line.rstrip() for line in f.readlines()[28:]]
	f.close()
	dssp = {int(line[0:5].strip()): {'pdb_num': line[5:11].strip(), 'pdb_chain': line[11:12].strip(), 
									 'pdb_resn': line[13].strip()} for line in lines}
	dssp = pd.DataFrame.from_dict(dssp, orient='index')
	return dssp
	
	

