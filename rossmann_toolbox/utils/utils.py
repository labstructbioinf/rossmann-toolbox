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
    sharp_probs = np.zeros(len(probs))
    for (beg, end, prob) in merged_peak_dict.values():
        sharp_probs[beg:end] = prob
    return sharp_probs

def corr_seq(seq):
    """
    Corrects sequence by mapping non-std residues to 'X'
    :param seq: input sequence
    :return: corrected sequence with non-std residues changed to 'X'
    """
    letters = set(list('ACDEFGHIKLMNPQRSTVWYX'))
    seq = ''.join([aa if aa in letters else 'X' for aa in seq])
    return seq

