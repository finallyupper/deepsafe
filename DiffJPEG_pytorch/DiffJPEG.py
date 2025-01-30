# Pytorch
import torch
import torch.nn as nn
# Local
from DiffJPEG_pytorch.compression import compress_jpeg
from DiffJPEG_pytorch.decompression import decompress_jpeg
from DiffJPEG_pytorch.utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round 
        self.rounding = rounding
        self.height = height
        self.width = width 
        self.factor = quality_to_factor(quality)
    #    self.compress = compress_jpeg(rounding=rounding, factor=factor)
    #    self.decompress = decompress_jpeg(height, width, rounding=rounding,factor=factor)

    def forward(self, x):
        y, cb, cr = compress_jpeg(x, self.rounding, self.factor) 
        recovered = decompress_jpeg(y, cb, cr, self.height, self.width, self.factor)
        # y, cb, cr = self.compress(x)
        # recovered = self.decompress(y, cb, cr)
        return recovered