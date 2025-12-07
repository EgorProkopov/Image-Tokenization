# TODO: rework wavelet tokenizer

import pywt
import torch
import numpy as np
import torch.nn as nn


class WaveletTokenizer(nn.Module):
    """
    Old and unaccurate wavelet tokenizer implementation

    SHOUDL BE REWORKED
    """
    def __init__(self, wavelet='haar', bit_planes=4, final_threshold=2**-3):
        super().__init__()
        self.wavelet = wavelet
        self.bit_planes = bit_planes
        self.final_threshold = final_threshold
        self.tokens = {
            "Group4x4": 0,
            "Group2x2": 1,
            "Insignificant": 2,
            "NowSignificantPos": 3,
            "NowSignificantNeg": 4,
            "NextAccuracy0": 5,
            "NextAccuracy1": 6
        }

        self.vocab_size = 7

    def forward(self, image: torch.Tensor):
        """
        Receives:
            image: Tensor [B, C, H, W] or [C, H, W] or [H, W]
        Outputs:
            Tensor [B, L]
        """
        # --- Случай батча ---
        if image.dim() == 4:
            B = image.size(0)
            tokens_list = [self.forward(image[b]) for b in range(B)]  # 1D-tensors
            lengths = [t.numel() for t in tokens_list]
            L = max(lengths)
            pad_idx = self.tokens["Insignificant"]

            # pads every sequence to L and configure into batch
            padded = []
            for t in tokens_list:
                if t.numel() < L:
                    pad = t.new_full((L - t.numel(),), pad_idx)
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)  # [B, L]
        
        # [H, W] → [1, H, W]; [C, H, W] → process each channel separately
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() != 3:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")
        C, H, W = image.shape

        tokens_sequence = []

        for c in range(C):
            channel = image[c].cpu().numpy()
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.bit_planes)
            coeff_arr, _ = pywt.coeffs_to_array(coeffs)  # [M, N]
            approx_coeffs = np.zeros_like(coeff_arr)

            max_abs = np.max(np.abs(coeff_arr))
            m_tilde = int(np.ceil(np.log2(max_abs))) if max_abs > 0 else 0
            T = 2 ** (m_tilde - 1)

            for _ in range(self.bit_planes):
                self._scan_bit_plane(coeff_arr, approx_coeffs, tokens_sequence, T)
                T /= 2
                if T < self.final_threshold:
                    break

        return torch.tensor(tokens_sequence, dtype=torch.long)

    def _scan_bit_plane(self, coeff_arr, approx_coeffs, tokens_sequence, T):
        M, N = coeff_arr.shape
        for i in range(0, M, 4):
            for j in range(0, N, 4):
                block = coeff_arr[i:i+4, j:j+4]
                if np.all(np.abs(block) < T):
                    tokens_sequence.append(self.tokens["Group4x4"])
                else:
                    for ii in range(i, min(i+4, M), 2):
                        for jj in range(j, min(j+4, N), 2):
                            sub = coeff_arr[ii:ii+2, jj:jj+2]
                            sub_approx = approx_coeffs[ii:ii+2, jj:jj+2]
                            if np.all(np.abs(sub) < T):
                                tokens_sequence.append(self.tokens["Group2x2"])
                            else:
                                for iii in range(ii, min(ii+2, M)):
                                    for jjj in range(jj, min(jj+2, N)):
                                        coef = coeff_arr[iii, jjj]
                                        approx = sub_approx[iii-ii, jjj-jj]
                                        if np.abs(coef) >= 2*T:
                                            tok = "NextAccuracy1" if np.abs(coef) > np.abs(approx) else "NextAccuracy0"
                                            tokens_sequence.append(self.tokens[tok])
                                            delta = T/4 if tok=="NextAccuracy1" else -T/4
                                            approx_coeffs[iii, jjj] += np.sign(coef)*delta
                                        elif T <= np.abs(coef) < 2*T:
                                            now = "NowSignificantPos" if coef>0 else "NowSignificantNeg"
                                            tokens_sequence.append(self.tokens[now])
                                            approx_coeffs[iii, jjj] = np.sign(coef)*1.5*T
                                            next_tok = "NextAccuracy1" if np.abs(coef)>1.5*T else "NextAccuracy0"
                                            tokens_sequence.append(self.tokens[next_tok])
                                            delta = T/4 if next_tok=="NextAccuracy1" else -T/4
                                            approx_coeffs[iii, jjj] += np.sign(coef)*delta
                                        else:
                                            tokens_sequence.append(self.tokens["Insignificant"])


