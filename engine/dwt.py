"""
Reproduced
"""
import torch
import pywt
import torch.nn as nn
from pytorch_wavelets import DWTForward  # Forward DWT
import numpy as np
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch
import cv2 

# def get_y_channels(data):
#         outputs = []
#         for x in data:
#             ycrcb_image = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)  # BGR -> YCrCb
#             y_channel = ycrcb_image[:, :, 0] 
#             y_tensor = torch.tensor(y_channel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#             outputs.append(y_tensor) 
#         return outputs 

def get_y_channels(data):
    outputs = []
    for x in data:
        # Assuming x is in the shape (C, H, W) and in RGB format
        if x.shape[0] != 3:
            raise ValueError("Input tensor must have 3 channels (RGB).")
        
        # Extract RGB channels
        R, G, B = x[0, :, :], x[1, :, :], x[2, :, :]

        # Compute Y channel using the formula for Y in YCrCb
        # Y = 0.299 * R + 0.587 * G + 0.114 * B
        Y = 0.299 * R + 0.587 * G + 0.114 * B

        # Convert Y to a tensor with shape (1, 1, H, W)
        y_tensor = Y.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions

        outputs.append(y_tensor)
    return outputs

def get_dwt(dwt, data):
                LL_lst = []
                for y in data:
                    yl, _ = dwt(y) 
                    LL_lst.append(yl) 
                return LL_lst 

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='haar', mode='zero'): 
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh


# class DWTForward_Init(nn.Module):
#     def __init__(self, J=1,  mode='zero', wave='haar'):
#         super(DWTForward_Init, self).__init__()
#         self.dwt = DWTForward(J=J, wave=wave)

#     def forward(self, x):
#         Yl, Yh = self.dwt(x)  # Yl: 저주파(LL), Yh: 고주파(LH, HL, HH)
#         return Yl, Yh

# class DWTForward_Init(nn.Module):
#     def __init__(self, J=1, mode='zero', wave='haar'):
#         super(DWTForward_Init, self).__init__()
#         self.J = J  
#         self.mode = mode
#         self.wave = wave

#     def forward(self, x):
#         coeffs = pywt.wavedec2(x.squeeze().cpu().numpy(), 
#                                self.wave, 
#                                mode=self.mode, 
#                                level=self.J)
#         # 저주파 성분(LL)과 고주파 성분(LH, HL, HH)을 분리
#         Yl = torch.tensor(coeffs[0]).unsqueeze(0).to(x.device)  # LL 성분
#         Yh = [torch.tensor(detail).unsqueeze(0).to(x.device) for detail in coeffs[1:]]  # LH, HL, HH 성분
#         return Yl, Yh

class DWTInverse_Init(nn.Module):
    def __init__(self, mode='zero', wave='haar'):
        super(DWTInverse_Init, self).__init__()
        self.mode = mode
        self.wave = wave

    def forward(self, Yl, Yh):
        # 역변환을 위해 저주파와 고주파 성분을 결합
        coeffs = [Yl.squeeze().cpu().numpy()] + [detail.squeeze().cpu().numpy() for detail in Yh]
        rec_image = pywt.waverec2(coeffs, self.wave, mode=self.mode)
        return torch.tensor(rec_image).unsqueeze(0).to(Yl.device)
