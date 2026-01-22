
# A non-official implementation of the Complex Ratio Mask (CRM) technique as a Pytorch module.

Implementation of the the Complex Ratio Mask (CRM) technique used in ["DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement"](https://arxiv.org/abs/2008.00264).

## Installlation
```bash
pip install CRM_pytorch
```
## Usage
```python
import torch
from CRM_pytorch import ComplexRatioMask

# Create instances of CRMLayer with every mode
crm = ComplexRatioMask(masking_mode='E') # Other modes include 'C' and 'R'. See paper for more information.

# Input parameters
batch_size = 32
time_steps = 100
freq_bins = 257

# Dummy input data (numpy first, then torch)
x_real = torch.randn(batch_size, time_steps, freq_bins, dtype=torch.float32)
x_imag = torch.randn(batch_size, time_steps, freq_bins, dtype=torch.float32)
mask_real = torch.ones(batch_size, time_steps, freq_bins, dtype=torch.float32)
mask_imag = torch.ones(batch_size, time_steps, freq_bins, dtype=torch.float32)

# Forward pass through the layer
estimated_speech = crm_layer_e(x_real, x_imag, mask_real, mask_imag)
```
