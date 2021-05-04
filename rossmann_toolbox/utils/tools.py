import sys
import subprocess

import atomium
sys.path.append(__file__)
from .solveX import solveX
from .dssp import parse_dssp_output, run_dssp

def run_command(cmd):
	cmd = 'LIBC_FATAL_STDERR_=1 ' + cmd # Suppress warnings of 'safe' crash in some FoldX calculation
	result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	return result.returncode

def add_wt_pdb_list(wt_seq, wt_pdb_list):
	wt_pdb_list=wt_pdb_list.copy()
    
	if wt_seq.find('-')==-1: return wt_pdb_list
	new_wt_pdb_list = []
	for pos, aa in enumerate(list(wt_seq)):

		if aa!='-':
			new_wt_pdb_list.append(wt_pdb_list.pop(0))
		else:
			new_wt_pdb_list.append(None)
	return new_wt_pdb_list

def generate_mutations(mutations_df, insertions=True):
	"""
	for pairs of wt_seq and mut_seq sequences detects mutations and encodes them
	according to the standard nomenclature
	"""
	res = []
	for idx, core in mutations_df.iterrows():
		mut_list = []
		
		for aa_idx in range(len(core.wt_seq)):
			wt_aa = core.wt_seq[aa_idx]
			mut_aa = core.mut_seq[aa_idx]
			wt_pos = core.wt_pdb_list[aa_idx]
			tmp=""
			if wt_pos == None and insertions: # insertion
				assert wt_aa == '-'
				tmp = f'{core.wt_seq[aa_idx-1]}{core.wt_pdb_list[aa_idx-1]}_{core.wt_seq[aa_idx+1]}{core.wt_pdb_list[aa_idx+1]}ins{mut_aa}'
			elif wt_aa != mut_aa: # deletion
				if mut_aa == '-':
					tmp = f'{wt_aa.upper()}{wt_pos}del'
				else: # substitution
					tmp = f'{wt_aa.upper()}{wt_pos}{mut_aa.upper()}'
			if tmp!="":mut_list.append(tmp)
		mut_str = ".".join(mut_list)
		res.append(mut_str)
	
	return res
	
	
def extract_core(pdb_path, chain, pdb_list, expected_seq):

	# check whether pdb numbering contains any negative numbers
	#if not len([i for i in pdb_list if i.lstrip("-").isnumeric() and int(i)<0])==0:
	#	warnings.warn("Atomium cannot handle negative values of residue ids!\nBe sure that you know what you're doing!\nhttps://github.com/samirelanduk/atomium/issues/29")

	# extract
	sel_res = [f"{chain}.{i}" for i in pdb_list]
	s = atomium.open(pdb_path)
	res = [i for i in s.model.residues() if i.id in sel_res]
	mymodel = atomium.Model(*[atomium.Chain(*res, id=chain)])
	
	# save
	out_file_name = pdb_path.rstrip('.pdb')+'.core.pdb'
	mymodel.save(out_file_name)
	
	# check	
	core_seq = "".join([mymodel.residue(i).code for i in sel_res])
	tmp=solveX(core_seq, expected_seq)
	assert tmp[1] == expected_seq, f'failed to save the core from {pdb_path}\n{core_seq}\n{expected_seq}'
	return out_file_name
	
def extract_core_dssp(template_pdb_file, wt_pdb_list, expected_seq, dssp_loc=None):
	dssp_data = run_dssp(template_pdb_file, dssp_bin=dssp_loc)
	wt_core_idx = dssp_data.pdb_num.isin(wt_pdb_list)
	wt_seq_dssp = "".join(dssp_data[wt_core_idx].pdb_resn) 
	if wt_seq_dssp.find("X")!=-1:    
		tmp=solveX(wt_seq_dssp, wt_seq_str)
		assert tmp[0]==0
		wt_seq_dssp = tmp[1]
	assert wt_seq_dssp == expected_seq
	wt_ss_dssp = "".join(dssp_data[wt_core_idx].pdb_ss)
	return wt_ss_dssp

