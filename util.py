import cv2
import numpy
import os
import torch
from torchvision import models

def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]


def load_images(image_paths, convert=None):
    iter_all_images = (cv2.resize(cv2.imread(fn), (256, 256)) for fn in image_paths)
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
    for i, image in enumerate(iter_all_images):
        if i == 0:
            all_images = numpy.empty((len(image_paths),) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = numpy.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
    ).reshape(new_shape)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience (int): 개선되지 않을 경우 몇 에포크까지 기다릴 것인지
            min_delta (float): 최소 손실 감소량 (이보다 작으면 개선되지 않았다고 간주)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss, model, checkpoint_path):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"EarlyStopping: New best loss found. Model saved at {checkpoint_path}")
        else:
            self.counter += 1
            print(f"EarlyStopping: No improvement for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                print("EarlyStopping: Stopping training.")
                return True
        return False
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval() 
        for param in vgg.parameters():
            param.requires_grad = False 
        self.vgg = vgg

    def forward(self, generated, target):
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        return F.mse_loss(gen_features, target_features)