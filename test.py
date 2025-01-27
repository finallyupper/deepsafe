import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from DiffJPEG_pytorch import DiffJPEG
from dataset import split_dataset, transform_test
from DualDefense_gan_fs import DualDefense 
from facenet_pytorch import InceptionResnetV1
import lpips
from engine.utils import set_seed, test_argument_parser, calculate_psnr, calculate_ssim, load_yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import torch.nn.functional as F 

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()  # 16층까지만 사용
        for param in vgg.parameters():
            param.requires_grad = False  # 모델의 가중치는 고정
        self.vgg = vgg

    def forward(self, generated, target):
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        return F.mse_loss(gen_features, target_features)
    
def load_model(device, type='vgg'):
    if type=='vgg':
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        resnet.classify = True
        resnet = resnet.to(device)
    return resnet  

def test(config, model, jpeg, test_loader_1, test_loader_2, device):
    message_size, save_path = config
    trump_true_size = 0; cage_true_size = 0 #NOTE:이게 뭐지...
    img_idx = 0

    model.encoder.eval()
    model.decoder.eval() 
    
    with torch.no_grad():
        for (trump_test_x, cage_test_x) in zip(test_loader_1, test_loader_2):
            trump_test_x = trump_test_x.to(device) 
            cage_test_x = cage_test_x.to(device) 

            trump_true_size = trump_true_size + len(trump_test_x)
            cage_true_size = cage_true_size + len(cage_test_x)

            trump_message = torch.randint(0, 2, (trump_test_x.shape[0], message_size), dtype=torch.float).to(device).detach()
            cage_message = torch.randint(0, 2, (cage_test_x.shape[0], message_size), dtype=torch.float).to(device).detach()

            encoded_trump = model.encode(trump_test_x, trump_message)
            encoded_cage = model.encode(cage_test_x, cage_message)

            # Compress encoded image with given quality degree
            encoded_trump = jpeg(encoded_trump).to(device)
            encoded_cage = jpeg(encoded_cage).to(device) 

            # Faceswap with Opponent
            # 1) FS in original image
            _, _, trump_df = model.deepfake1(trump_test_x, 'B')
            _, _, cage_df = model.deepfake1(cage_test_x, 'A')
            # 2) FS in encoded image
            _, _, encoded_trump_df = model.deepfake1(encoded_trump, 'B') # returns x, logits, x
            _, _, encoded_cage_df = model.deepfake1(encoded_cage, 'A')

            for k in range(len(trump_test_x)):
                cv2.imwrite(os.path.join(save_path, 'img/real', str(img_idx) + '_' + str(k) + '_A.png'),(trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode', str(img_idx) + '_' + str(k) + '_A.png'),(encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode_fake', str(img_idx) + '_' + str(k) + '_A.png'),(encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/fake', str(img_idx) + '_' + str(k) + '_A.png'),(trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                
                trump_psnr_sum += calculate_psnr((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_ssim_sum += calculate_ssim((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_psnr_sum += calculate_psnr((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_ssim_sum += calculate_ssim((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())


            for k in range(len(cage_test_x)):
                cv2.imwrite(os.path.join(save_path, 'img/real', str(img_idx) + '_' + str(k) + '_B.png'),(cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode', str(img_idx) + '_' + str(k) + '_B.png'),(encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode_fake', str(img_idx) + '_' + str(k) + '_B.png'),(encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/fake', str(img_idx) + '_' + str(k) + '_B.png'),(cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                
                cage_psnr_sum += calculate_psnr((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_ssim_sum += calculate_ssim((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_psnr_sum += calculate_psnr((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_ssim_sum += calculate_ssim((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
  

def main():
    set_seed(0)
    args = test_argument_parser() 
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')

    # Lpips
    loss_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    loss_vgg = lpips.LPIPS(net='vgg').to(device)  # closer to "traditional" perceptual loss

    height, width = 160, 160 
    test_config = load_yaml(args.config_path)['test'] 
    trump_path = test_config['trump_path'] 
    cage_path = test_config['cage_path'] 
    model_path = test_config['model_path'] 
    save_path = test_config['save_path']
    quality = test_config['quality']

    print(f'Results will be save at {save_path}') 
    os.makedirs(save_path, exist_ok=True) 
    save_result = ['encode', 'encode_fake', 'fake', 'real']
    for path in save_result: os.makedirs(os.path.join(save_path, 'img', path), exist_ok=True)

    print(f'Load model from {model_path}') 
    model = DualDefense(test_config['message_size'],in_channels=3,device=device) 
    model.encoder.load_state_dict(torch.load(model_path)['encoder'], strict=False)
    model.decoder.load_state_dict(torch.load(model_path)['decoder'], strict=False)

    model.encoder.eval()
    model.decoder.eval()

    print('Load classifier') 
    resnet = load_model(device, type='vgg') 

    print('Split & Load dataset') 
    _, _, trump_test_dataset = split_dataset(trump_path, test_transform=transform_test, val_ratio=0, test_ratio=1)
    _, _, cage_test_dataset = split_dataset(cage_path, test_transform=transform_test, val_ratio=0, test_ratio=1) 
    trump_test_loader = DataLoader(trump_test_dataset, batch_size=args.batch_size, shuffle=False)
    cage_test_loader = DataLoader(cage_test_dataset, batch_size=args.batch_size, shuffle=False)

    assert len(trump_test_loader) == len(cage_test_loader), "The pair data size should be same"
    jpeg = DiffJPEG.DiffJPEG(
        height=height, width=width,
        differentiable=True,
        quality=quality
    ).to(device) 
    test(trump_test_loader, cage_test_loader,
         )



if __name__ == "__main__":
    main()
