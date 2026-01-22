import torch
import torch.nn as nn

class ComplexRatioMask(nn.Module):
    def __init__(self, masking_mode: str):
        super().__init__()
        self.masking_mode = masking_mode
        self.eps = 1e-8

        valid_modes = ['E', 'C', 'R']  # Taken from paper
        if masking_mode not in valid_modes:
            raise ValueError(
                f"Invalid masking_mode: {masking_mode}. Must be one of {valid_modes}"
            )

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        mask_real: torch.Tensor,
        mask_imag: torch.Tensor,
        return_real_imag: bool = False,
    ):
        """
        Inputs:
         - x_real: real component of noisy signal
         - x_imag: imaginary component of noisy signal
         - mask_real: real component of mask
         - mask_imag: imaginary component of mask

        Returns:
         - estimated_speech: complex-valued estimated speech
        """

        if self.masking_mode == 'E':
            # Magnitude and phase of noisy signal
            x_mag = torch.sqrt(x_real**2 + x_imag**2)
            x_phase = torch.atan2(x_imag, x_real)

            # Magnitude and phase of mask
            mask_real = mask_real + self.eps
            mask_mag = torch.sqrt(mask_real**2 + mask_imag**2)
            mask_mag = torch.tanh(mask_mag)

            mask_phase = torch.atan2(mask_imag, mask_real)

            # Apply mask
            est_mags = mask_mag * x_mag
            est_phase = x_phase + mask_phase

            est_real = est_mags * torch.cos(est_phase)
            est_imag = est_mags * torch.sin(est_phase)

        elif self.masking_mode == 'C':
            # Complex mask
            est_real = x_real * mask_real - x_imag * mask_imag
            est_imag = x_real * mask_imag + x_imag * mask_real

        else:  # 'R'
            # Real mask
            est_real = x_real * mask_real
            est_imag = x_imag * mask_imag

        if return_real_imag:
            return est_real, est_imag
        else:
            return torch.complex(est_real, est_imag)
