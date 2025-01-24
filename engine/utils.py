import math 
import cv2 
import random 
import torch 
import argparse
import numpy as np 
import yaml 
import pywt 
from facenet_pytorch import MTCNN, InceptionResnetV1 
import torch.nn.functional as F

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_model(device, type='vgg'):
    if type=='vgg':
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        resnet.classify = True
        resnet = resnet.to(device)
    #    mtcnn = MTCNN(image_size=160).to(device) 
    return resnet

def get_ckpt_name(config):
    ckpt_best = f'ckpt_best_lam{config["lambda_val"]}_al{config["alpha_val"]}.pt' 
    ckpt_img_best = f'ckpt_best_img_lam{config["lambda_val"]}_al{config["alpha_val"]}.pt' 
    ckpt_final = f'ckpt_final_lam{config["lambda_val"]}_al{config["alpha_val"]}.pt' 
    return ckpt_best, ckpt_img_best, ckpt_final 

def default_argument_parser():
    parser = argparse.ArgumentParser(description='TAW')
    parser.add_argument('--config_path', default="/home/yoojinoh/Others/PR/deepfake-free/DualDefense/config.yaml", type=str)
    parser.add_argument('--save_path',  type=str)
    parser.add_argument('--gpus', default="0")
    # parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    # parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    # parser.add_argument('--epoch', default=2500, type=int, help='epochs')
    # parser.add_argument('--start_decode', default=30, type=int, help='epoch to start training decoder')
    # parser.add_argument('--clip', default=15, type=int, help='clip')
    # parser.add_argument('--message_size', default=15, type=int, help='msg size')

    # parser.add_argument('--lambda_val', default=2, type=float, help='weight of msg loss') #Beta in paper Modified(Yoojin)
    # parser.add_argument('--alpha_val', default=0.5, type=float, help='weight of image loss') #Modified(Yoojin)

    # parser.add_argument('--T_max', default=50, type=int, help='cosine annealing LR scheduler t_max')
    # parser.add_argument('--name', default='ckpt-new', type=str, help='name to save')
    # parser.add_argument('--gpus', default='1', type=str, help='id of gpus to use')
    return parser.parse_args()

def test_argument_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config_path', default="/home/yoojinoh/Others/PR/deepfake-free/DualDefense/config.yaml", type=str) #ADD(Yoojin)
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--message_size', default=15, type=int, help='msg size')
    parser.add_argument('--gpus', default='1', type=str, help='id of gpus to use')
    return parser.parse_args()

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# separate the ycbcr images for the loss calculation
# def ycbcr_images(cover_ybr,stego_ybr,device):
#     dwtimage = DWTForward_Init(J=1, mode='zero', wave='haar').to(device)
#     #idwtimage = DWTInverse_Init(mode='zero', wave='haar').to(device)
#     stego_Yl, stego_Yh = dwtimage(stego_ybr[:, 0, :, :].unsqueeze(1).to(device))
#     cover_Yl, cover_Yh = dwtimage(cover_ybr[:, 0, :, :].unsqueeze(1).to(device))
#     stego_color = torch.cat((stego_ybr[:, 1, :, :], stego_ybr[:, 2, :, :]), 1).unsqueeze(1)
#     cover_color = torch.cat((cover_ybr[:, 1, :, :], cover_ybr[:, 2, :, :]), 1).unsqueeze(1)
#     stego_YH = torch.tensor([item.cpu().detach().numpy() for item in stego_Yh]).to(device).squeeze()
#     cover_YH = torch.tensor([item.cpu().detach().numpy() for item in cover_Yh]).to(device).squeeze()

#     return stego_Yl, stego_Yh, cover_Yl, cover_Yh, stego_color, cover_color, stego_YH, cover_YH

#Add(Yoojin)
def custom_ssim_loss(original_image, watermarked_image, L=1.0, k1=0.01, k2=0.03):
    """
    Calculate SSIM-based loss for the low-frequency sub-bands of the original and watermarked images.
    
    Args:
        original_image: Tensor of shape (Batch, Channels, Height, Width), normalized to [0, 1].
        watermarked_image: Tensor of shape (Batch, Channels, Height, Width), normalized to [0, 1].
        L: Dynamic range of the pixel values (default: 1.0 for normalized inputs).
        k1: Stabilizing constant for mean (default: 0.01).
        k2: Stabilizing constant for variance (default: 0.03).

    Returns:
        ssim_loss: Average SSIM loss across the batch.
    """
    # Constants for stability
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Compute means using a Gaussian kernel
    window_size = 11
    sigma = 1.5
    channels = original_image.size(1)
    
    # Create Gaussian kernel
    kernel = torch.tensor(
        [[torch.exp(-(x**2 + y**2) / (2 * sigma**2)) for x in range(-(window_size // 2), window_size // 2 + 1)]
         for y in range(-(window_size // 2), window_size // 2 + 1)],
        dtype=original_image.dtype, device=original_image.device
    )
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)

    # Apply Gaussian kernel
    mu_O = F.conv2d(original_image, kernel, padding=window_size // 2, groups=channels)
    mu_W = F.conv2d(watermarked_image, kernel, padding=window_size // 2, groups=channels)

    mu_O_sq = mu_O ** 2
    mu_W_sq = mu_W ** 2
    mu_O_mu_W = mu_O * mu_W

    sigma_O = F.conv2d(original_image * original_image, kernel, padding=window_size // 2, groups=channels) - mu_O_sq
    sigma_W = F.conv2d(watermarked_image * watermarked_image, kernel, padding=window_size // 2, groups=channels) - mu_W_sq
    sigma_OW = F.conv2d(original_image * watermarked_image, kernel, padding=window_size // 2, groups=channels) - mu_O_mu_W

    # Compute SSIM
    numerator = (2 * mu_O_mu_W + c1) * (2 * sigma_OW + c2)
    denominator = (mu_O_sq + mu_W_sq + c1) * (sigma_O + sigma_W + c2)
    ssim_map = numerator / (denominator + 1e-7)

    # Compute SSIM loss
    ssim_loss = 1 - ssim_map.mean()  # Average across spatial dimensions and batch
    return ssim_loss

def perform_dwt(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0]
    coeffs = pywt.dwt2(y_channel, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

#From Dual Defense(https://github.com/Xming-zzz/DualDefense/blob/master/train.py)
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

#From Dual Defense(https://github.com/Xming-zzz/DualDefense/blob/master/train.py)
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input ids must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
def calculate_ssim_torch(img1, img2, window_size=11, channel=1):
    """
    Compute SSIM using PyTorch tensors.
    Assumes img1 and img2 are tensors of shape (N, 1, H, W).
    """
    K1, K2 = 0.01, 0.03
    C1 = (K1 * 255) ** 2
    C2 = (K2 * 255) ** 2

    # Gaussian window
    def create_window(window_size, channel):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        _1D_window = torch.exp(-coords**2 / (2 * 1.5**2))
        _1D_window /= _1D_window.sum()
        _2D_window = _1D_window[:, None] * _1D_window[None, :]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

# def calculate_ssim_loss(original_image, watermarked_image):
#     """
#     Calculate the SSIM loss between the low-frequency sub-bands of the original and watermarked images.
#     Args:
#         original_image: Original image in BGR format.
#         watermarked_image: Watermarked image in BGR format.
#     Returns:
#         SSIM loss.
#     """
#     batch_size = original_image.size(0)
#     # original_image = original_image.permute(0, 2, 3, 1)  # Change to (Batch, Height, Width, Channel)
#     # watermarked_image = watermarked_image.permute(0, 2, 3, 1) 
#     ssim_losses = []

#     for i in range(batch_size):
#         LL = original_image[i].detach().cpu().numpy() # (1, 80, 80)
#         LL_prime = watermarked_image[i].detach().cpu().numpy()

#         # Calculate SSIM for a single pair
#         ssim = calculate_ssim(LL[0], LL_prime[0])  # Implement SSIM based on the equation
#         ssim_loss = 1 - ssim
#         ssim_losses.append(ssim_loss)

#     # Average loss across the batch
#     avg_ssim_loss = np.mean(np.array(ssim_losses))
#     return avg_ssim_loss


def salt_noise(SNR, img_tensor, device):
    mask_noise = np.random.choice((0, 1, 2), size=(1, HW, HW), p=[SNR, (1 - SNR) / 2, (1 - SNR) / 2])
    mask_noise = np.repeat(mask_noise, 3, axis=0)
    # mask_noise = np.expand_dims(mask_noise, 0).repeat(len(img_tensor), axis=0)
    img_np = img_tensor.cpu().numpy()
    for img_np_i in img_np:
        img_np_i[mask_noise == 1] = 255
        img_np_i[mask_noise == 2] = 0
    return torch.from_numpy(img_np).to(device)


def sp_noise(img_tensor, prob):
    # Salt-pepper noise    
    p_1 = prob / 2 # prob_1 = 1 - prob
    p_2 = 1 - p_1
    l_l = int(img_tensor.shape[0])
    c = int(img_tensor.shape[1])
    h_l = int(img_tensor.shape[2])
    w_l = int(img_tensor.shape[3])
    for num in range(l_l):
        for h in range(h_l):
            for w in range(w_l):
                rdn = random.random()
                if rdn < p_1:
                    img_tensor[num, :, h, w] = 0
                elif rdn > p_2:
                    img_tensor[num, :, h, w] = 1  # 255
    return img_tensor


def gaussian_noise(image, mean=0, var=0.001):
    # Adds Gaussian Noise
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    out = cv2.filter2D(image, -1, kernel=kernel)
    return out


def MedianFilter(img_tensor, k_size):
    img_np = img_tensor.permute(0, 2, 3, 1).cpu().numpy()
    for idx in range(len(img_np)):
        img = cv2.medianBlur(img_np[idx], k_size)
        img = img.transpose(2, 0, 1)
        img_tensor[idx] = torch.from_numpy(img).data
    return img_tensor 

