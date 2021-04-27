![logo](https://github.com/labstructbioinf/rossmann-toolbox/blob/main/logo.png?raw=true)

![python-ver](https://img.shields.io/badge/python-%3E=3.6.1-blue)
[![codecov](https://codecov.io/gh/labstructbioinf/rossmann-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/labstructbioinf/rossmann-toolbox)

<b> Prediction and re-engineering of the cofactor specificity of Rossmann-fold proteins</b>

### Installation

```
pip install rossmann-toolbox
```

Alternatively, to get the most recent changes, install directly from the repository:
```
pip install git+https://github.com/labstructbioinf/rossmann-toolbox.git
```

#### For some of the features additional dependencies are required:
- [<b>FoldX4</b>](http://foldxsuite.crg.eu/)
- [<b>DSSP3</b>](https://github.com/cmbi/dssp)
- [<b>HH-suite3</b>](https://github.com/soedinglab/hh-suite)


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
#### What next?
To learn about other features of the `rossmann-toolbox`, such as <b>structure-based prediction</b> and <b>visualization of the results</b>, please refer to the notebook `examples/example_minimal.ipynb`. 

### Contact
If you have any questions, problems or suggestions, please contact us.  The `rossmann-toolbox` was developed by Kamil Kaminski, Jan Ludwiczak, Maciej Jasinski, Adriana Bukala, 
Rafal Madaj, Krzysztof Szczepaniak, and Stanislaw Dunin-Horkawicz.

This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.
