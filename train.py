# author:oldpan
# data:2018-4-16
# Just for study and research

from __future__ import print_function
import argparse
import os

import cv2
import numpy as np
import torch

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from models import Autoencoder, toTensor, var_to_np
from util import get_image_paths, load_images, stack_images, EarlyStopping
from training_data import get_training_data
import matplotlib.pyplot as plt 
from tqdm import tqdm  
from torch import optim

parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--gpus', type=int, default=0, metavar='N')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda is True:
    print('===> Using GPU to train')
    device = torch.device(f'cuda:{args.gpus}')
    cudnn.benchmark = True
else:
    print('===> Using CPU to train')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('===> Loading datasets')
images_A = get_image_paths("/data1/yoojinoh/def/new_data/faceDataset2/byeonFace")
images_B = get_image_paths("/data1/yoojinoh/def/new_data/faceDataset2/chaFace")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))


model = Autoencoder().to(device)

start_epoch = 0
print('===> Start from scratch')
criterion = nn.L1Loss()
perceptual_loss = nn.MSELoss()#Add(Yoojin)
alpha = 1.0  # L1 loss weight
beta = 0.01  # perceptual loss weight

lr_list_1 = []
lr_list_2 = []

#Modified(Yoojin)
    
optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_A.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_B.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))

schedule_config = {
    'milestones':[1000, 4000, 8000, 12000],
    'gamma' : 0.5,
}
scheduler_1 = optim.lr_scheduler.MultiStepLR(optimizer_1, **schedule_config)
scheduler_2 = optim.lr_scheduler.MultiStepLR(optimizer_2, **schedule_config)
#early_stopping = EarlyStopping(patience=200, min_delta=0.0001)
min_loss = np.inf 

if __name__ == "__main__":
    save_root = args.save_dir
    checkpoint_dir = os.path.join(save_root, 'checkpoint')
    outputs = os.path.join(save_root, 'train_outputs')
    os.makedirs(save_root, exist_ok = True) 
    os.makedirs(checkpoint_dir, exist_ok = True) 
    os.makedirs(outputs, exist_ok=True) 
    lossA_lst = []; lossB_lst = [] 
    
    print('Start training, press \'q\' to stop')
    for epoch in range(start_epoch, args.epochs):
        batch_size = args.batch_size

        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)

        warped_A, target_A = toTensor(warped_A), toTensor(target_A)
        warped_B, target_B = toTensor(warped_B), toTensor(target_B)

        if args.cuda:
            warped_A = warped_A.to(device).float()
            target_A = target_A.to(device).float()
            warped_B = warped_B.to(device).float()
            target_B = target_B.to(device).float()

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        warped_A = model(warped_A, 'A')
        warped_B = model(warped_B, 'B')

        #Modified(Yoojin)
        loss1 = alpha * criterion(warped_A, target_A) + beta * perceptual_loss(warped_A, target_A)
        loss2 = alpha * criterion(warped_B, target_B) + beta * perceptual_loss(warped_B, target_B)

        loss = loss1.item() + loss2.item()
        loss1.backward()
        loss2.backward()

        optimizer_1.step()
        optimizer_2.step()
        scheduler_1.step()
        scheduler_2.step()
        lr_list_1.append(scheduler_1.get_last_lr()[0])
        lr_list_2.append(scheduler_2.get_last_lr()[0])

        print('epoch: {}, lossA:{}, lossB:{}'.format(epoch, loss1.item(), loss2.item()))
        lossA_lst.append(loss1.item()) 
        lossB_lst.append(loss2.item()) 
        if epoch % 1000 == 0:
            print(f"[DEBUG]>>Epoch {epoch}, LR: {scheduler_1.get_last_lr()[0]:.6f}")
        if epoch % args.log_interval == 0:
            test_A_ = target_A[0:14]
            test_B_ = target_B[0:14]
            test_A = var_to_np(target_A[0:14])
            test_B = var_to_np(target_B[0:14])
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state, os.path.join(checkpoint_dir, f'bc_ckpt_e{epoch}.t7'))

            plt.figure(figsize=(10, 5))
            plt.plot(range(len(lossA_lst)), lossA_lst, label='Loss A')
            plt.plot(range(len(lossB_lst)), lossB_lst, label='Loss B')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(outputs, 'loss_plot.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(range(len(lr_list_1)), lr_list_1, label='Learning Rate 1')
            plt.plot(range(len(lr_list_2)), lr_list_2, label='Learning Rate 2')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(outputs, 'lr_plot.png'))
            plt.close()

        if loss < min_loss:
            min_loss = loss
            test_A_ = target_A[0:14]
            test_B_ = target_B[0:14]
            test_A = var_to_np(target_A[0:14])
            test_B = var_to_np(target_B[0:14])
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state, os.path.join(checkpoint_dir, f'bc_ckpt_best.t7'))

        figure_A = np.stack([
            test_A,
            var_to_np(model(test_A_, 'A')),
            var_to_np(model(test_A_, 'B')),
        ], axis=1)
        figure_B = np.stack([
            test_B,
            var_to_np(model(test_B_, 'B')),
            var_to_np(model(test_B_, 'A')),
        ], axis=1)

        figure = np.concatenate([figure_A, figure_B], axis=0)
        figure = figure.transpose((0, 1, 3, 4, 2))
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        figure = np.clip(figure * 255, 0, 255).astype('uint8')

        cv2.imwrite(os.path.join(outputs, 'output.png'), figure)
        
        # if early_stopping(loss1.item() + loss2.item(), model, os.path.join(checkpoint_dir, 'bc_ckpt_best.t7')):
        #     break

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(lr_list_1)), lr_list_1, label='Learning Rate 1')
    plt.plot(range(len(lr_list_2)), lr_list_2, label='Learning Rate 2')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outputs, 'final_lr_plot.png'))
    plt.close()