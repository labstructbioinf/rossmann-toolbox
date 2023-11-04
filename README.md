# Rossmann Toolbox

<img src="https://github.com/labstructbioinf/rossmann-toolbox/blob/main/logo.png" align="center">

The Rossmann Toolbox provides two deep learning models for predicting the cofactor specificity of Rossmann enzymes based on either the sequence or the structure of the beta-alpha-beta cofactor binding motif.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#Usage)
    + [Sequence-based approach](#sequence-based-approach)
    + [Structure-based approach](#structure-based-approach)
    + [EGATConv layer](#EGATConv-layer)
* [ Remarks ](#Remarks)
    + [How to cite](#how-to-cite)
    + [Contact](#contact)
    + [Funding](#funding)

# Installation

Create a conda environment:
```bash
conda create --name rtb python=3.6.2
conda activate rtb
```

Install pip in the environment:
```bash
conda install pip
```

Install rtb using `requirements.txt`:
```bash
pip install -r requirements.txt
```

# Usage

## Sequence-based approach
The input is a full-length sequence. The algorithm first detects <b>Rossmann cores</b> (i.e. the β-α-β motifs that interact with the cofactor) in the sequence and later evaluates their cofactor specificity:
```python
import matplotlib.pylab as plt
from rossmann_toolbox import RossmannToolbox
rtb = RossmannToolbox(use_gpu=True)

# Eample 1
# The b-a-b core is predicted in the full-length sequence

data = {'3m6i_A': 'MASSASKTNIGVFTNPQHDLWISEASPSLESVQKGEELKEGEVTVAVRSTGICGSDVHFWKHGCIGPMIVECDHVLGHESAGEVIAVHPSVKSIKVGDRVAIEPQVICNACEPCLTGRYNGCERVDFLSTPPVPGLLRRYVNHPAVWCHKIGNMSYENGAMLEPLSVALAGLQRAGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGRLKFAKEICPEVVTHKVERLSAEESAKKIVESFGGIEPAVALECTGVESSIAAAIWAVKFGGKVFVIGVGKNEIQIPFMRASVREVDLQFQYRYCNTWPRAIRLVENGLVDLTRLVTHRFPLEDALKAFETASDPKTGAIKVQIQSLE'}

preds = rtb.predict(data, mode='seq', core_detect_mode='dl', importance=False)

# Eample 2
# The b-a-b cores are provided by the user (WT vs mutant)

data = {'seq_wt': 'AGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGR', # WT, binds NAD
        'seq_mut': 'AGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITSRDEGR'} # D211S, I212R mutant, binds NADP

preds, imps = rtb.predict(data, mode='core', importance=True)

# Example 3
# Which residues contributed most to the prediction of WT as NAD-binding?
seq_len = len(data['seq_wt'])
plt.errorbar(list(range(1, seq_len+1)),
             imps['seq_wt']['NAD'][0], yerr=imps['seq_wt']['NAD'][1], ecolor='grey')

```

For more examples of how to use the sequence-based approach, see [example_minimal.ipynb](https://github.com/labstructbioinf/rossmann-toolbox/blob/main/examples/example_minimal.ipynb).

## Structure-based approach
Structure-based predictions are not currently available. We are working on a new version that will not only provide predictions, but also the ability to make specificity-shifting mutations.

## EGATConv layer

The structure-based predictor includes an EGAT layer that deals with graph neural networks supporting edge features. The EGAT layer is available from DGL, and you can find more details about it in the [DGL documentation](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.EGATConv.html). For a detailed description of the EGAT layer and its usage, please refer to the supplementary materials of the [Rossmann Toolbox paper](https://academic.oup.com/bib/article/23/1/bbab371/6375059).

# Remarks
## How to cite?
If you find the `rossmann-toolbox` useful, please cite the paper:

*Rossmann-toolbox: a deep learning-based protocol for the prediction and design of cofactor specificity in Rossmann fold proteins*
Kamil Kamiński, Jan Ludwiczak, Maciej Jasiński, Adriana Bukala, Rafal Madaj, Krzysztof Szczepaniak, Stanisław Dunin-Horkawicz
*Briefings in Bioinformatics*, Volume 23, Issue 1, January 2022, [bbab371](https://doi.org/10.1093/bib/bbab371)

## Contact
If you have any questions, problems or suggestions, please contact [us](https://ibe.biol.uw.edu.pl/en/835-2/research-groups/laboratory-of-structural-bioinformatics/).

## Funding
This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.
