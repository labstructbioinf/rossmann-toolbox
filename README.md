# Rossmann Toolbox

<img src="https://github.com/labstructbioinf/rossmann-toolbox/blob/main/logo.png" align="center">

The Rossmann Toolbox provides two deep learning models for predicting the cofactor specificity of Rossmann enzymes based on either the sequence or the structure of the beta-alpha-beta cofactor binding motif.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#Usage)
    + [Sequence-based approach](#sequence-based-approach)
    + [Structure-based approach](#structure-based-approach)
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
from rossmann_toolbox import RossmannToolbox
rtb = RossmannToolbox(use_gpu=True)

data = {'3m6i_A': 'MASSASKTNIGVFTNPQHDLWISEASPSLESVQKGEELKEGEVTVAVRSTGICGSDVHFWKHGCIGPMIVECDHVLGHESAGEVIAVHPSVKSIKVGDRVAIEPQVICNACEPCLTGRYNGCERVDFLSTPPVPGLLRRYVNHPAVWCHKIGNMSYENGAMLEPLSVALAGLQRAGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGRLKFAKEICPEVVTHKVERLSAEESAKKIVESFGGIEPAVALECTGVESSIAAAIWAVKFGGKVFVIGVGKNEIQIPFMRASVREVDLQFQYRYCNTWPRAIRLVENGLVDLTRLVTHRFPLEDALKAFETASDPKTGAIKVQIQSLE'}

preds = rtb.predict(data, mode='seq')
preds = {'3m6i_A': {'FAD': 0.0008955444,
                    'NAD': 0.998446,
                    'NADP': 0.00015508455,
                    'SAM': 0.0002544397, ...}}
```

## Structure-based approach
Structure-based predictions are not currently available. We are working on a new version that will not only provide predictions, but also the ability to make specificity-shifting mutations.

# Remarks

## How to cite?
If you find the `rossmann-toolbox` useful, please cite the preprint:

"*Rossmann-toolbox: a deep learning-based protocol for the prediction and design of cofactor specificity in Rossmann-fold proteins*"
by Kamil Kaminski, Jan Ludwiczak, Maciej Jasinski, Adriana Bukala, Rafal Madaj, Krzysztof Szczepaniak, and Stanislaw Dunin-Horkawicz
bioRxiv 2021.05.05.440912; doi: https://doi.org/10.1101/2021.05.05.440912

## Contact
If you have any questions, problems or suggestions, please contact [us](https://ibe.biol.uw.edu.pl/en/835-2/research-groups/laboratory-of-structural-bioinformatics/).

## Funding
This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.
