'''
storage for dictionaris and other variable that may cause 
potential problems
'''

LABEL_DICT = {'NAD' : 0, 'NADP' : 1, 'SAM' : 2,  'FAD' : 3}
LABEL_DICT_R = {0 :'NAD', 1 :'NADP', 2 :'SAM',  3:'FAD' }

SS_MAP_EXT = {
    'H' : 0,
    'H2' : 0,
    'B' : 1,
    'G' : 2,
    'I' : 3,
    'T' : 4,
    'S' : 5,
    '-' : 6,
    'C' : 6,
    ' ' : 6,
    'E1' : 7,
    'E2' : 8,
    '?' : 6,
    'H1' : 9,
    'H3' : 10
}



ACIDS_ORDER = 'ARNDCQEGHILKMFPSTWYVX'

ACIDS_MAP_DEF = {acid : nb for  nb, acid in enumerate(ACIDS_ORDER)}
ACIDS_MAP_R = {nb : acid for  nb, acid in enumerate(ACIDS_ORDER)}

CM_THRESHOLD = 7

NUM_SS_LETTERS = len(set(SS_MAP_EXT.values()))
NUM_RES_LETTERS = len(set(ACIDS_ORDER))