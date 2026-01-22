import torch
import os
import sys

# Add parent directory to path (same logic as TF test)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CRM_pytorch import ComplexRatioMask


def test_ComplexRatioMask():
    # Define the input shape and create some dummy data
    batch_size = 32
    time_steps = 100
    freq_bins = 257  # Complex input: real and imaginary parts

    # Dummy input data (numpy first, then torch)
    x_real = torch.randn(batch_size, time_steps, freq_bins, dtype=torch.float32)
    x_imag = torch.randn(batch_size, time_steps, freq_bins, dtype=torch.float32)
    mask_real = torch.ones(batch_size, time_steps, freq_bins, dtype=torch.float32)
    mask_imag = torch.ones(batch_size, time_steps, freq_bins, dtype=torch.float32)

    # Create instances of CRMLayer with every mode
    crm_layer_e = ComplexRatioMask(masking_mode='E')
    crm_layer_c = ComplexRatioMask(masking_mode='C')
    crm_layer_r = ComplexRatioMask(masking_mode='R')

    # Forward pass through the layers
    estimated_speech = crm_layer_e(x_real, x_imag, mask_real, mask_imag)
    assert estimated_speech.shape == (batch_size, time_steps, freq_bins)
    assert torch.is_complex(estimated_speech)

    estimated_speech = crm_layer_c(x_real, x_imag, mask_real, mask_imag)
    assert estimated_speech.shape == (batch_size, time_steps, freq_bins)
    assert torch.is_complex(estimated_speech)

    estimated_speech = crm_layer_r(x_real, x_imag, mask_real, mask_imag)
    assert estimated_speech.shape == (batch_size, time_steps, freq_bins)
    assert torch.is_complex(estimated_speech)

    # Print a summary of the test
    print("\nComplexRatioMask (PyTorch) test passed successfully.")


if __name__ == "__main__":
    test_ComplexRatioMask()
