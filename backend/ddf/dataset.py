import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision.transforms import transforms
from faceswap_pytorch.models import Autoencoder, toTensor, var_to_np
from faceswap_pytorch.training_data import get_training_data 
from faceswap_pytorch.umeyama import * 
import torch 

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation((-10, 10)),
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)), ### ADDED(YOOJIN) 
])

class CustomDataset(Dataset):
    def __init__(self, images, transform=None, type='train'):
        self.images = images
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = Image.open(self.ids[idx])
        image = Image.fromarray(cv2.imread(self.images[idx]))

        # image = cv2.imread(self.ids[idx])

        if self.type == "train":
            if self.transform:
                image = self.transform(image)
                # image, _ = random_warp(image)#Added(Yoojin)
                # if isinstance(image, np.ndarray):  # If random_warp returned a numpy array
                #     image = torch.from_numpy(image).permute(2, 0, 1) 
        if self.type=="test":
            if self.transform:
                image = self.transform(image) 
        return image


def split_dataset(path, train_transform=None, valid_transform=None, test_transform=None, val_ratio=0.2, test_ratio=0.2):
    images = [x.path for x in os.scandir(path) if x.name.endswith(".jpg") or x.name.endswith(".png")]
    images = [os.path.join(root, file) for root,_,files in os.walk(path) for file in files if file.endswith(".jpg") or file.endswith(".png")]
    total_len = len(images)
    train_images = images[: int(total_len * (1 - val_ratio - test_ratio))]
    val_images = images[int(total_len * (1 - val_ratio - test_ratio)): int(total_len * (1 - test_ratio))]
    test_images = images[int(total_len * (1 - test_ratio)):]
    print(f"# of Train:{len(train_images)} | Valid: {len(val_images)} | Test: {len(test_images)}")
    return CustomDataset(train_images, train_transform), CustomDataset(val_images, valid_transform), CustomDataset(test_images, test_transform, "test")


'''
trans_test = transforms.Compose([
    transforms.ToTensor()
])

background = cv2.imread('')
cv2.imwrite('tmp1.png', background)
yuv_background = cv2.cvtColor(background, cv2.COLOR_BGR2YUV)  # 
Y, U, V = yuv_background[..., 0], yuv_background[..., 1], yuv_background[..., 2]
YY = trans_test(Y)
YYY = YY.permute(1, 2, 0).squeeze().numpy() * 255
print(yuv_background[..., 0] == YYY)
x = yuv_background[..., 0] == YYY

yuv_background[..., 0] = YYY
mm = cv2.cvtColor(yuv_background, cv2.COLOR_YUV2BGR)
cv2.imwrite('tmp2.png', mm)

'''

# get pair of random warped images from aligened face image
def random_warp(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(1, 2, 0).cpu().numpy() 
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - 80, 128 + 80, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)

    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    target_image = cv2.warpAffine(image, mat, (64, 64))

    return warped_image, target_image
