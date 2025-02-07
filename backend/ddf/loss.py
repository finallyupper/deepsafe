import torch
import torch.nn as nn 

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
    
class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = c.w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)