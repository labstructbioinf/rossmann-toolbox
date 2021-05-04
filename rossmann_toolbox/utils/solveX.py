import numpy as np
import itertools

class Found(Exception):
	pass

def find_index(s, ch):
	"""
	solveX helper function
	"""

	if type(s)==str:
		a = np.array(list(s))
	elif type(s)==list:
		a = np.asarray(s)
	else:
		print ("solveX, find_index, unexpected input type")
		sys.exit(-1)
	
	return list(np.where(a == ch)[0])

def solveX(small_seq, big_seq, can_be_X = 'MCKH'):
	"""
	Finds position of small_seq in big_seq even if small_seq contains "X" (unknown residues)
	:param small_seq: aa sequence
	:param big_seq: aa sequence
	:param can_be_X: aa that can be X in small_seq
	:return: position of small_seq in big_seq (-1 if not found)
	"""
	
	indexes = find_index(small_seq, 'X')

	# There is one ore more "X" in small_seq
	if len(indexes)>0: 

		try:
			for perm in itertools.product(can_be_X, repeat=len(indexes)):

				subseq_alt = small_seq
				for jpos, j in enumerate(indexes):
					subseq_alt = subseq_alt[:j] + perm[jpos] + subseq_alt[j+1:]

				subseqpos = big_seq.find(subseq_alt)
				if subseqpos!=-1: 
					raise Found
		
		except Found:
			return big_seq.find(subseq_alt), subseq_alt
		else:
			return -1, small_seq
	else:
		return big_seq.find(small_seq), small_seq 
		
		
def solveX_rev(small_seq, big_seq, can_be_X = 'MCKH'):
	"""
	Finds position of small_seq in big_seq even if big_seq contains "X" (unknown residues)
	:param small_seq: aa sequence
	:param big_seq: aa sequence
	:param can_be_X: aa that can be X in small_seq
	:return: [position of small_seq in big_seq (-1 if not found), small_seq]
	"""
	
	indexes = find_index(big_seq, 'X')

	# There is one ore more "X" in small_seq
	if len(indexes)>0: 

		try:
			for perm in itertools.product(can_be_X, repeat=len(indexes)):

				subseq_alt = big_seq
				for jpos, j in enumerate(indexes):
					subseq_alt = subseq_alt[:j] + perm[jpos] + subseq_alt[j+1:]

				subseqpos = subseq_alt.find(small_seq)
				if subseqpos!=-1: 
					raise Found
		
		except Found:
			return subseq_alt.find(small_seq), small_seq
		else:
			return -1, small_seq
	else:
		return big_seq.find(small_seq), small_seq 