import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from DiffJPEG_pytorch import DiffJPEG
from dataset import split_dataset, transform_test
from DualDefense_gan_fs import DualDefense as DualDefense_160
from facenet_pytorch import InceptionResnetV1
import lpips
from engine.utils import set_seed, test_argument_parser, calculate_psnr, calculate_ssim, load_yaml
import warnings
warnings.filterwarnings('ignore')

def test_process_vgg(trump_test_loader, cage_test_loader, args, save_path, loss_alex, loss_vgg, model, jpeg, resnet, min_label, max_label, device):
    """JPEG CASE (Attack way = 0)"""
    test_message_correct, test_df_message_correct, test_size = 0, 0, 0
    trump_psnr_sum, cage_psnr_sum, trump_ssim_sum, cage_ssim_sum = 0, 0, 0, 0
    trump_df_psnr_sum, cage_df_psnr_sum, trump_df_ssim_sum, cage_df_ssim_sum = 0, 0, 0, 0
    trump_suc, cage_suc, trump_suc_ori, cage_suc_ori = 0, 0, 0, 0
    trump_suc_real, cage_suc_real, trump_attack_success, cage_attack_success = 0, 0, 0, 0
    attack_l1_trump_sum, attack_l1_cage_sum, attack_l2_trump_sum, attack_l2_cage_sum = 0, 0, 0, 0
    trump_true_size, cage_true_size = 0, 0
    lpips_alex_trump, lpips_alex_cage, lpips_vgg_trump, lpips_vgg_cage = 0, 0, 0, 0
    cage_dist, trump_dist = 0, 0
    img_i = 0
    with torch.no_grad():
        for (trump_test_x, cage_test_x) in zip(trump_test_loader, cage_test_loader):
            trump_test_x = trump_test_x.to(device)
            cage_test_x = cage_test_x.to(device)

            trump_true_size = trump_true_size + len(trump_test_x)
            cage_true_size = cage_true_size + len(cage_test_x)

            trump_message = torch.randint(0, 2, (trump_test_x.shape[0], args.message_size), dtype=torch.float).to(device).detach()
            cage_message = torch.randint(0, 2, (cage_test_x.shape[0], args.message_size), dtype=torch.float).to(device).detach()
            
            encoded_trump = model.encode(trump_test_x, trump_message)
            encoded_cage = model.encode(cage_test_x, cage_message)
            encoded_trump = jpeg(encoded_trump).to(device)
            encoded_cage = jpeg(encoded_cage).to(device)


            _, _, encoded_trump_df = model.deepfake1(encoded_trump, 'B') # returns x, logits, x
            _, _, encoded_cage_df = model.deepfake1(encoded_cage, 'A')
            _, _, trump_df = model.deepfake1(trump_test_x, 'B')
            _, _, cage_df = model.deepfake1(cage_test_x, 'A')


            for k in range(len(trump_test_x)):
                cv2.imwrite(os.path.join(save_path, 'img/real', str(img_i) + '_' + str(k) + '_A.png'),(trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode', str(img_i) + '_' + str(k) + '_A.png'),(encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode_fake', str(img_i) + '_' + str(k) + '_A.png'),(encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/fake', str(img_i) + '_' + str(k) + '_A.png'),(trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                
                trump_psnr_sum += calculate_psnr((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_ssim_sum += calculate_ssim((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_psnr_sum += calculate_psnr((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_ssim_sum += calculate_ssim((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),(encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())

            for k in range(len(cage_test_x)):
                cv2.imwrite(os.path.join(save_path, 'img/real', str(img_i) + '_' + str(k) + '_B.png'),(cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode', str(img_i) + '_' + str(k) + '_B.png'),(encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/encode_fake', str(img_i) + '_' + str(k) + '_B.png'),(encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(save_path, 'img/fake', str(img_i) + '_' + str(k) + '_B.png'),(cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                
                cage_psnr_sum += calculate_psnr((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_ssim_sum += calculate_ssim((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_psnr_sum += calculate_psnr((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_ssim_sum += calculate_ssim((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                
            
            for k in range(len(trump_df)):
                mask = abs(trump_df[k] - encoded_trump_df[k])
                mask = mask[0, :, :] + mask[1, :, :] + mask[2, :, :]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if (((trump_df[k] * mask - encoded_trump_df[k] * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                    trump_dist += 1

                attack_l1 = torch.nn.functional.l1_loss(trump_df[k], encoded_trump_df[k])
                attack_l2 = torch.nn.functional.mse_loss(trump_df[k], encoded_trump_df[k])
                attack_l1_trump_sum = attack_l1_trump_sum + attack_l1
                attack_l2_trump_sum = attack_l2_trump_sum + attack_l2
                lpips_alex_trump = lpips_alex_trump + loss_alex(trump_df[k].unsqueeze(0),
                                                                encoded_trump_df[k].unsqueeze(0))
                lpips_vgg_trump = lpips_vgg_trump + loss_vgg(trump_df[k].unsqueeze(0), encoded_trump_df[k].unsqueeze(0))
                trump_attack_success = trump_attack_success + 1 if attack_l2 >= 0.05 else trump_attack_success

            for k in range(len(cage_df)):
                mask = abs(cage_df[k] - encoded_cage_df[k])
                mask = mask[0, :, :] + mask[1, :, :] + mask[2, :, :]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if (((cage_df[k] * mask - encoded_cage_df[k] * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                    cage_dist += 1

                attack_l1 = torch.nn.functional.l1_loss(cage_df[k], encoded_cage_df[k])
                attack_l2 = torch.nn.functional.mse_loss(cage_df[k], encoded_cage_df[k])
                attack_l1_cage_sum = attack_l1_cage_sum + attack_l1
                attack_l2_cage_sum = attack_l2_cage_sum + attack_l2
                lpips_alex_cage = lpips_alex_cage + loss_alex(cage_df[k].unsqueeze(0), encoded_cage_df[k].unsqueeze(0))
                lpips_vgg_cage = lpips_vgg_cage + loss_vgg(cage_df[k].unsqueeze(0), encoded_cage_df[k].unsqueeze(0))
                cage_attack_success = cage_attack_success + 1 if attack_l2 >= 0.05 else cage_attack_success

            img_i += 1

            encoded_trump_df_message = model.decode(encoded_trump_df)
            encoded_cage_df_message = model.decode(encoded_cage_df)
            encoded_trump_message = model.decode(encoded_trump)
            encoded_cage_message = model.decode(encoded_cage)

            test_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + ((encoded_cage_df_message > 0.5) == cage_message).sum().item()
            test_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + ((encoded_cage_message > 0.5) == cage_message).sum().item()

            trump_test_x_160 = trump_test_x.clone()
            img_probs = resnet(trump_test_x_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == min_label:
                    trump_suc_real = trump_suc_real + 1


            cage_test_x_160 = cage_test_x.clone()
            img_probs = resnet(cage_test_x_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)

            for i in range(len(label_tensor)):
                if label_tensor[i] == max_label:
                    cage_suc_real = cage_suc_real + 1

            encoded_trump_df_160 = encoded_trump_df.clone()
            img_probs = resnet(encoded_trump_df_160) 
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)

            for i in range(len(label_tensor)):
                if label_tensor[i] != max_label:
                    trump_suc = trump_suc + 1
            encoded_cage_df_160 = encoded_cage_df.clone()
            img_probs = resnet(encoded_cage_df_160)  
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)

            for i in range(len(label_tensor)):
                if label_tensor[i] != min_label:
                    cage_suc = cage_suc + 1
            trump_df_160 = trump_df.clone()
            img_probs = resnet(trump_df_160) 
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)

            for i in range(len(label_tensor)):
                if label_tensor[i] == max_label:
                    trump_suc_ori = trump_suc_ori + 1
            cage_df_160 = cage_df.clone()
            img_probs = resnet(cage_df_160)  
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)

            for i in range(len(label_tensor)):
                if label_tensor[i] == min_label:
                    cage_suc_ori = cage_suc_ori + 1

            test_size += trump_test_x.shape[0] + cage_test_x.shape[0]

        test_message_acc = test_message_correct / (test_size * args.message_size)
        test_df_message_acc = test_df_message_correct / (test_size * args.message_size)
        trump_psnr_avg = trump_psnr_sum / trump_true_size
        cage_psnr_avg = cage_psnr_sum / cage_true_size
        trump_ssim_avg = trump_ssim_sum / trump_true_size
        cage_ssim_avg = cage_ssim_sum / cage_true_size
        trump_df_psnr_avg = trump_df_psnr_sum / trump_true_size
        cage_df_psnr_avg = cage_df_psnr_sum / cage_true_size
        trump_df_ssim_avg = trump_df_ssim_sum / trump_true_size
        cage_df_ssim_avg = cage_df_ssim_sum / cage_true_size
        trump_suc_p = trump_suc / trump_true_size
        cage_suc_p = cage_suc / cage_true_size
        trump_suc_p_ori = trump_suc_ori / trump_true_size
        cage_suc_p_ori = cage_suc_ori / cage_true_size
        trump_suc_p_real = trump_suc_real / trump_true_size
        cage_suc_p_real = cage_suc_real / cage_true_size
        trump_attack_success_p = trump_attack_success / trump_true_size
        cage_attack_success_p = cage_attack_success / cage_true_size
        attack_l1_trump_res = attack_l1_trump_sum / trump_true_size
        attack_l1_cage_res = attack_l1_cage_sum / cage_true_size
        attack_l2_trump_res = attack_l2_trump_sum / trump_true_size
        attack_l2_cage_res = attack_l2_cage_sum / cage_true_size
        lpips_alex_trump_res = lpips_alex_trump / trump_true_size
        lpips_alex_cage_res = lpips_alex_cage / cage_true_size
        lpips_vgg_trump_res = lpips_vgg_trump / trump_true_size
        lpips_vgg_cage_res = lpips_vgg_cage / cage_true_size
        trump_dist_mask = trump_dist / trump_true_size
        cage_dist_mask = cage_dist / cage_true_size

        f_print_vgg = open(os.path.join(save_path, 'log.txt'), "w")
        print(f"Trump PSNR : {trump_psnr_avg}, Cage PSNR : {cage_psnr_avg}\n",
              f"Trump SSIM : {trump_ssim_avg}, Cage SSIM : {cage_ssim_avg}\n",
              f"Trump_df PSNR : {trump_df_psnr_avg}, Cage_df PSNR : {cage_df_psnr_avg}\n",
              f"Trump_df SSIM : {trump_df_ssim_avg}, Cage_df SSIM : {cage_df_ssim_avg}\n",
              f"Test encoded msg accuracy : {test_message_acc}, Test DF msg accuracy : {test_df_message_acc}\n",
              f"enADFtoX---trump : {trump_suc_p}, cage : {cage_suc_p}\n",
              f"ADFtoB---trump : {trump_suc_p_ori}, cage : {cage_suc_p_ori}\n",
              f"AtoA---trump : {trump_suc_p_real}, cage : {cage_suc_p_real}\n",
              f"l2 suc---trump : {trump_attack_success_p}, cage : {cage_attack_success_p}\n",
              f"l1---trump : {attack_l1_trump_res}, cage : {attack_l1_cage_res}\n",
              f"l2---trump : {attack_l2_trump_res}, cage : {attack_l2_cage_res}\n",
              f"lpips_alex---trump : {lpips_alex_trump_res.item()}, cage : {lpips_alex_cage_res.item()}\n",
              f"lpips_vgg---trump : {lpips_vgg_trump_res.item()}, cage : {lpips_vgg_cage_res.item()}\n",
              f"mask---trump : {trump_dist_mask}, cage : {cage_dist_mask}\n",
              file=f_print_vgg)
        f_print_vgg.close()

def load_model(device, type='vgg'):
    if type=='vgg':
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        resnet.classify = True
        resnet = resnet.to(device)
    return resnet  

def main():
    set_seed(0)
    args = test_argument_parser()
    print(args) 
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')
    loss_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    loss_vgg = lpips.LPIPS(net='vgg').to(device)  # closer to "traditional" perceptual loss
    HW = 160
    min_label = 38; max_label = 397
    config = load_yaml('config.yaml')['test'] 
    trump_path = config['trump_path'] 
    cage_path = config['cage_path'] 
    model_path = config['model_path'] 
    save_path = config['save_path']
    os.makedirs(save_path, exist_ok=True) 
    save_result = ['encode', 'encode_fake', 'fake', 'real', 'val', 'tmp']
    for path in save_result:
        os.makedirs(os.path.join(save_path, 'img', path), exist_ok=True)

    # lfw 
    model = DualDefense_160(args.message_size, in_channels=3, device=device)
    model.encoder.load_state_dict(torch.load(model_path)['encoder'], strict=False)
    model.decoder.load_state_dict(torch.load(model_path)['decoder'], strict=False)
    model.encoder.eval()
    model.decoder.eval()

    resnet = load_model(device)

    _, _, trump_test_dataset = split_dataset(trump_path, test_transform=transform_test, val_ratio=0, test_ratio=1)
    _, _, cage_test_dataset = split_dataset(cage_path, test_transform=transform_test, val_ratio=0, test_ratio=1)

    trump_test_loader = DataLoader(trump_test_dataset, batch_size=args.batch_size, shuffle=False)
    cage_test_loader = DataLoader(cage_test_dataset, batch_size=args.batch_size, shuffle=False)
    quality_list = [50]

    for quality in quality_list:
        quality = quality
        jpeg = DiffJPEG.DiffJPEG(height=HW, width=HW, differentiable=True, quality=quality).to(device)
        test_process_vgg(trump_test_loader, cage_test_loader,
                         args,save_path,
                         loss_alex, loss_vgg, 
                         model, 
                         jpeg, 
                         resnet,
                         min_label, max_label,
                         device) 

if __name__ == "__main__":
    main()
