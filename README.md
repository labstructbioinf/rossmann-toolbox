![logo](https://github.com/labstructbioinf/rossmann-toolbox/blob/main/logo.png?raw=true)

![python-ver](https://img.shields.io/badge/python-%3E=3.6.1-blue)
[![codecov](https://codecov.io/gh/labstructbioinf/rossmann-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/labstructbioinf/rossmann-toolbox)

<b> Prediction and re-engineering of the cofactor specificity of Rossmann-fold proteins</b>

### Installation

#### Before you start

The `rossmann-toolbox` can be accessed via a web server available at (https://lbs.cent.uw.edu.pl/rossmann-toolbox)

#### Instructions

```
pip install rossmann-toolbox
```

Alternatively, to get the most recent changes, install directly from the repository:
```
pip install git+https://github.com/labstructbioinf/rossmann-toolbox.git
```

#### For some of the features additional dependencies are required:
| Package                                       | Sequence variant | Structure variant |
|-----------------------------------------------|:----------------:|:-----------------:|
|[**FoldX4**](http://foldxsuite.crg.eu/)        | -                | **required**      |
|[**DSSP3**](https://github.com/cmbi/dssp)      | -                | **required**      |
|[**HH-suite3**](https://github.com/soedinglab/hh-suite) | optional| optional          |

### Getting started

#### Sequence-based approach
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

#### Structure-based approach
The input is a protein structure. Preparation steps are the same as above, but additionally, structural features are calculated via **FOLDX** software, and secondary structure features via **DSSP**
```python
# required binaries
PATH_FOLDX = ...
PATH_HHPRED = ...
PATH_DSSP = ...

path_to_structures = ...  # path to pdb files
chains_to_use = ... # chains to load from `path_to_structures`
rtb = RossmannToolbox(use_gpu=False, foldx_loc = PATH_FOLDX, 
                                     hhsearch_loc = PATH_HHPRED,
                                     dssp_loc = PATH_DSSP)

preds = rtb.predict_structure(path_to_structures, chains_to_use, mode='seq', core_detect_mode='dl')
preds = [{'NAD': 0.99977881,
  'NADP': 0.0018195,
  'SAM': 0.00341983,
  'FAD': 3.62e-05,
  'seq': 'AGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGRL',
  'NAD_std': 0.0003879,
  'NADP_std': 0.00213571,
  'SAM_std': 0.00411747,
  'FAD_std': 3.95e-05}]
```

#### What next?
To learn about other features of the `rossmann-toolbox`, such as <b>visualization of the results</b>, please refer to the notebook `examples/example_minimal.ipynb`. 

### Remarks

#### How to cite?
If you find the `rossmann-toolbox` useful, please cite the preprint:

"*Graph neural networks and sequence embeddings enable the prediction and design of the cofactor specificity of Rossmann fold proteins*"
by Kamil Kaminski, Jan Ludwiczak, Maciej Jasinski, Adriana Bukala, Rafal Madaj, Krzysztof Szczepaniak, and Stanislaw Dunin-Horkawicz
bioRxiv 2021.05.05.440912; doi: https://doi.org/10.1101/2021.05.05.440912

#### Contact
If you have any questions, problems or suggestions, please contact [us](https://lbs.cent.uw.edu.pl).

#### Funding
This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.
