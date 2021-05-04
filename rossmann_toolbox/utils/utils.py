from collections.abc import Iterable

import numpy as np
import warnings
from scipy.signal import find_peaks
from itertools import groupby


def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f'{filename}:{lineno} - {message}')

def sharpen_preds(probs, sep=15, min_prob=0.5):
    """
    Sharpens raw probabilities to more human-readable format
    :param probs: raw probabilities
    :return: sharpened probabilities
    """
    probs = probs.flatten()
    above_threshold = probs > min_prob
    peak_dict = {}
    i = 0
    for k, g in groupby(enumerate(above_threshold), key=lambda x: x[1]):
        if k:
            g = list(g)
            beg, end = g[0][0], g[-1][0]
            if end -beg >= 1:
                peak_dict[i] = (beg, end, max(probs[beg:end]))
                i += 1
    if len(peak_dict) == 1:
        merged_peaks = [list(peak_dict.keys())]
    else:
        merged_peaks = []
        i = 0
        while i <= len(peak_dict) -1:
            merge_list = [i]
            while True:
                if peak_dict[i + 1][0] - peak_dict[i][1] <= sep:
                    merge_list.append(i + 1)
                    i += 1
                    if i == len(peak_dict) - 1:
                        merged_peaks.append(merge_list)
                        break
                else:
                    merged_peaks.append(merge_list)
                    i += 1
                    break
            if i == len(peak_dict) - 1:
                break

    merged_peak_dict = {i: (peak_dict[mp[0]][0], peak_dict[mp[-1]][1], max([peak_dict[idx][2] for idx in mp]))
                        for i, mp in enumerate(merged_peaks)}
    return merged_peak_dict

def corr_seq(seq):
    """
    Corrects sequence by mapping non-std residues to 'X'
    :param seq: input sequence
    :return: corrected sequence with non-std residues changed to 'X'
    """
    letters = set(list('ACDEFGHIKLMNPQRSTVWYX'))
    seq = ''.join([aa if aa in letters else 'X' for aa in seq])
    return seq


def separate_beta_helix(secondary):
    '''
    changes labels of helices in dssp sequences, adding number for them for instance:
    `H-E1-H-E2-H` to `'H1-E1-H2-E2-H3'
    :params iterable with chars indicating dssp annotations
    "return listo of chars enhanced dssp annotations"
    '''
    if isinstance(secondary, str):
        secondary = list(secondary)
    elif not isinstance(secondary, Iterable):
        raise ValueError(f'secondary must be iterable, but passed {type(secondary)}')
        
    sec_len = len(secondary)
    #E1 E2 split condition
    h_start = sec_len//2
    #H split condition
    e_indices = [i for i, letter in enumerate(secondary)  if letter =='E']
    e_min, e_max = min(e_indices), max(e_indices)
    secondary_extended = list()
    for i, letter in enumerate(secondary):
        if letter == 'E':
            if i <= h_start:
                new_letter = 'E1'
            else:
                new_letter = 'E2'
        elif letter == 'H':
            if i < e_min:
                new_letter = 'H1'
            elif e_min < i < e_max:
                new_letter = 'H2'
            else:
                new_letter = 'H3'
        elif letter == ' ':
            new_letter = '-'
        else:
            new_letter = letter
        secondary_extended.append(new_letter)
    return secondary_extended


