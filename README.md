# Rossmann Toolbox

## Cookbook
### Init
```python
from rossmann_toolbox import RossmannToolbox
rtb = RossmannToolbox(use_gpu=True)
```
### Evaluate cores

```python
data = {'seq1': 'MSKKFNGKVCLVTGAGGNIGLATALRLAEEGTAIALLDMNREAL', 
        'seq2': 'MSKKFNGKVCLVTGAGGNIGLATALRLAEEGTAIALLSRNREAL'}
preds = rtb.predict(data, mode='core', importance=False)
preds = {'seq1': {'FAD': 0.0010036804,
  'FAD_std': 0.001190269,
  'NAD': 0.9867387,
  'NAD_std': 0.016175654,
  'NADP': 0.014890989,
  'NADP_std': 0.015133685,
  'SAM': 0.00017169576,
  'SAM_std': 0.0002028175},
 'seq2': {'FAD': 6.141185e-08,
  'FAD_std': 5.1703925e-08,
  'NAD': 1.699253e-05,
  'NAD_std': 2.501946e-05,
  'NADP': 0.9999896,
  'NADP_std': 1.4234308e-05,
  'SAM': 1.1083341e-05,
  'SAM_std': 1.9881409e-05}}
```

### Evaluate full-length sequences
```python
data = {'1piw_A': 'MSYPEKFEGIAIQSHEDWKNPKKTKYDPKPFYDHDIDIKIEACGVCGSDIHCAAGHWGNMKMPLVVGHEIVGKVVKLGPKSNSGLKVGQRVGVGAQVFSCLECDRCKNDNEPYCTKFVTTYSQPYEDGYVSQGGYANYVRVHEHFVVPIPENIPSHLAAPLLCGGLTVYSPLVRNGCGPGKKVGIVGLGGIGSMGTLISKAMGAETYVISRSSRKREDAMKMGADHYIATLEEGDWGEKYFDTFDLIVVCASSLTDIDFNIMPKAMKVGGRIVSISIPEQHEMLSLKPYGLKAVSISYSALGSIKELNQLLKLVSEKDIKIWVETLPVGEAGVHEAFERMEKGDVRYRFTLVGYDKEFSD'}
preds = rtb.predict(data, mode='seq', importance=False)

preds = {'1piw_A': {'FAD': 2.0404239e-11,
  'FAD_std': 3.5465863e-11,
  'NAD': 7.502697e-09,
  'NAD_std': 7.654336e-09,
  'NADP': 1.0,
  'NADP_std': 0.0,
  'SAM': 8.859513e-09,
  'SAM_std': 1.7599064e-08,
  'sequence': 'NGCGPGKKVGIVGLGGIGSMGTLISKAMGAETYVISRSSRKR'}}
```

## What next?

To learn about other features of the `rossmann-toolbox`, such as structure-based prediction and results visualization, please refer to `example_minimal.ipynb`. 

If you have any questions, problems or suggestions, please contact us. 
