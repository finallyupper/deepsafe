import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dataset import split_dataset 
from DualDefense_gan_fs import DualDefense 
from dataset import transform_train, transform_test, transform_val
from engine.utils import default_argument_parser, set_seed
from engine.utils import load_yaml, get_ckpt_name
import shutil 
import torch.nn as nn
import engine.utils 

def train(configs, 
          model, 
          train_loader_1, train_loader_2, val_loader_1, val_loader_2,
          optimizer, opt_discriminator, 
          lr_scheduler, 
          criterion, message_criterion, adversarial_loss, 
          ckpt_names, 
          device):
    
    min_val_loss = float('inf')
    min_val_image_loss = float('inf')
    epochs, message_size, d_iter, clip, save_path, name, beta, alpha = configs 
    lambda_en, lambda_s, lambda_adv = 0.9, 0.0, 0
    print(f'[INFO] alpha={alpha}, beta={beta}, clip={clip}, message_size={message_size}, d_iter={d_iter}')
    print(f'[INFO] Lambda of enc, ssim, adv is given as {lambda_en}, {lambda_s}, {lambda_adv}')
    ckpt_best, ckpt_img_best, ckpt_final = ckpt_names

    train_loss_plot = []
    train_ed_loss = []
    train_adv_loss = []
    train_gan_loss_plot = []
    # train_ssim_loss_plot = []

    val_loss_plot = []
    val_acc_plot = []
    val_df_acc_plot = []
    val_ed_loss = []
    val_adv_loss = []

    for epoch in range(epochs): 
        print(f'[INFO] Epoch {epoch + 1}/{epochs}')
        train_gan_loss=0 # Discriminator Loss 
        train_image_loss=0 #image quality loss
        train_image_df_loss=0#original domain attack loss
        train_message_loss=0 # message loss * beta(=2)
        train_message_correct, train_df_message_correct = 0, 0
        # train_ssim_loss=0
        val_image_loss, val_image_df_loss = 0, 0
        val_message_loss, val_message_correct, val_df_message_correct = 0, 0, 0 
        train_size, val_size = 0, 0 

        # Training mode
        model.encoder.train()
        model.decoder.train()
        model.adv_model.train()
        
        totlen = min(len(train_loader_1), len(train_loader_2))
        for trump_train_x, cage_train_x, in tqdm(zip(train_loader_1, train_loader_2), desc="Processing", total=totlen):
            trump_train_x = trump_train_x.to(device) 
            cage_train_x = cage_train_x.to(device)

            # Real=1, Fake=0
            trump_valid = Variable(torch.Tensor(len(trump_train_x), 1).to(device).fill_(1.0),requires_grad=False) # No Watermark image(=1)
            trump_fake = Variable(torch.Tensor(len(trump_train_x), 1).to(device).fill_(0.0), requires_grad=False) # Watermarked image(=0)
            cage_valid = Variable(torch.Tensor(len(cage_train_x), 1).to(device).fill_(1.0),requires_grad=False)
            cage_fake = Variable(torch.Tensor(len(cage_train_x), 1).to(device).fill_(0.0),requires_grad=False)

            # Define binary message
            trump_message = torch.randint(0, 2, (trump_train_x.shape[0], message_size), dtype=torch.float).to(device).detach()
            cage_message = torch.randint(0, 2, (cage_train_x.shape[0], message_size), dtype=torch.float).to(device).detach()

            #NOTE: About Discriminator 
            # Encode images with messages (Line 3)
            encoded_trump = model.encode(trump_train_x, trump_message)
            encoded_cage = model.encode(cage_train_x, cage_message) 

            opt_discriminator.zero_grad() 
            # Predict real/fake with Discriminator (Line 6)
            trump_pred_real = model.adv(trump_train_x.float().to(device))
            trump_pred_fake = model.adv(encoded_trump.float().to(device))
            cage_pred_real = model.adv(cage_train_x.float().to(device))
            cage_pred_fake = model.adv(encoded_cage.float().to(device))

            trump_discriminator_loss = adversarial_loss(trump_pred_real, trump_valid.to(device)) \
                + adversarial_loss(trump_pred_fake, trump_fake.to(device))
            cage_discriminator_loss = adversarial_loss(cage_pred_real, cage_valid.to(device)) \
                + adversarial_loss(cage_pred_fake, cage_fake.to(device))
            
            # Compute Discriminator loss LD (Line 7, Eq.8)
            discriminator_loss = (trump_discriminator_loss + cage_discriminator_loss)/2
            train_gan_loss += discriminator_loss.item() 

            # Update discriminator (Line 8)
            discriminator_loss.backward()
            opt_discriminator.step()
            
            # ------------------------------------------------------------------------------------------------------------------------ #
            #NOTE: About Face Swap (New gradient updates)
            optimizer.zero_grad()
            # Encode images with messages (Line 3)
            encoded_trump = model.encode(trump_train_x, trump_message)
            encoded_cage = model.encode(cage_train_x, cage_message) 

            # 1. Face Swap with Target User (Line 4)
            x1_trump, logits_original_trump, encoded_trump_original = \
                model.deepfake1(encoded_trump, 'A')
            x2_cage, logits_original_cage, encoded_cage_original = \
                model.deepfake1(encoded_cage, 'B')
            
            # 2. Face Swap with Source User (Face swap with opponent decoder) (Line 5)
            x1_trump_endf, logits_df_trump, encoded_trump_df = \
                model.deepfake1(encoded_trump, 'B')
            x2_cage_endf, logits_df_cage, encoded_cage_df = \
                model.deepfake1(encoded_cage, 'A')
            
            # 3. Compute Image Loss
            # 1)Image quality loss 2) Structural info compensation loss , 3) Original domain attack loss
            # L_img = lamb_en * L_en + lamb_s*L_s + lamb_adv*L_adv

            # 3-1. Image quality loss
            # Predict real/fake with encoded image by discriminator
            #CHECK(Yoojin): Expectation, Mean
            encoded_loss = criterion(encoded_trump, trump_train_x) + criterion(encoded_cage, cage_train_x) # Eq13, MSE Loss
            encoded_loss /= 2

            encoded_trump_pred = model.adv(encoded_trump) 
            encoded_cage_pred = model.adv(encoded_cage)

            encoded_discriminator_loss = adversarial_loss(encoded_trump_pred, trump_valid.to(device)) \
                + adversarial_loss(encoded_cage_pred, cage_valid.to(device)) 
            encoded_discriminator_loss /= 2
            image_encoded_loss = encoded_loss + encoded_discriminator_loss 

            # 3-3. Original domain attack loss
            # MSE Loss btw swap images (Eq 10, 11, 12)
            image_adv_logits_loss = (criterion(logits_original_trump, logits_df_trump) + \
                criterion(logits_original_cage, logits_df_cage)) # / 2 #CHECK: Divice 2?
            image_adv_logits_loss /= 2 
            
            image_loss = lambda_en * image_encoded_loss  + lambda_adv * image_adv_logits_loss  # + lambda_s * ssim_loss
            
            #Modified(Yoojin): No alpha multiplied
            loss = image_loss 
            train_image_loss += image_encoded_loss.item()
            train_image_df_loss += image_adv_logits_loss.item() 
            # train_ssim_loss += ssim_loss.item() #Add(Yoojin)

            #CHECK: torch.autograd.set_detect_anomaly(True) 

            if epoch >= d_iter:
                # Decode encoded image with encoded adv image
                encoded_trump_pred_message = model.decode(encoded_trump)
                encoded_cage_pred_message = model.decode(encoded_cage)

                encoded_trump_df_pred_message = model.decode(encoded_trump_df)
                encoded_cage_df_pred_message = model.decode(encoded_cage_df)

                # Compute Message Loss (Line 12)
                L_wm_en =  message_criterion(encoded_trump_pred_message, trump_message) + message_criterion(encoded_cage_pred_message, cage_message) 
                L_wm_adv = message_criterion(encoded_trump_df_pred_message, trump_message) + message_criterion(encoded_cage_df_pred_message, cage_message) 
                L_wm_en /= 2; L_wm_adv /=2
                message_loss = L_wm_en + L_wm_adv 

                train_df_message_correct += ((encoded_trump_df_pred_message > 0.5) == trump_message).sum().item() + \
                    ((encoded_cage_df_pred_message > 0.5) == cage_message).sum().item()
                train_message_correct += ((encoded_trump_pred_message > 0.5) == trump_message).sum().item() + \
                    ((encoded_cage_pred_message > 0.5) == cage_message).sum().item()

                #Modified(Yoojin): no beta multiplied
                train_message_loss += message_loss.item()

                # total loss
                loss *= alpha # image loss 
                loss += message_loss * beta
                # Update watermark encoder and decoder
            # Else: update watermark encoder 

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss.backward()
            optimizer.step()
            train_size += trump_train_x.shape[0] + cage_train_x.shape[0]
    
        # End of for loop
        # train_image_loss /= train_size
        # train_image_df_loss /= train_size
        # train_gan_loss /= train_size 
        # train_message_loss /= train_size
        # train_ssim_loss /= train_size

        train_loss_plot.append(train_message_loss)
        train_ed_loss.append(train_image_loss)
        train_adv_loss.append(train_image_df_loss)
        train_gan_loss_plot.append(train_gan_loss) 
        # train_ssim_loss_plot.append(train_ssim_loss)

        train_df_message_acc = train_df_message_correct / (train_size * message_size)
        train_message_acc = train_message_correct / (train_size * message_size)

        lr_scheduler.step()

        # Evaluation
        model.encoder.eval()
        model.decoder.eval() 

        with torch.no_grad():
            for (trump_val_x, cage_val_x) in zip(val_loader_1, val_loader_2):
                trump_val_x = trump_val_x.to(device)
                cage_val_x = cage_val_x.to(device)
                
                trump_message = torch.randint(0, 2, (trump_val_x.shape[0], message_size), dtype=torch.float).to(device).detach()
                cage_message = torch.randint(0, 2, (cage_val_x.shape[0], message_size), dtype=torch.float).to(device).detach()

                # Encode images
                encoded_trump = model.encode(trump_val_x, trump_message)
                encoded_cage = model.encode(cage_val_x, cage_message) 

                # Predict original swapped images 
                x1_original_trump_val, logits_original_trump_val, encoded_trump_original = \
                    model.deepfake1(encoded_trump, 'A')
                x2_original_cage_val, logits_original_cage_val, encoded_cage_original =  \
                    model.deepfake1(encoded_cage, 'B')
                # Predict opposite swapped images
                x1_val_endf_trump, logits_trump_endf_val, encoded_trump_df = \
                    model.deepfake1(encoded_trump, 'B')
                x2_val_endf_cage, logits_cage_endf_val, encoded_cage_df = \
                    model.deepfake1(encoded_cage, 'A')
                
                #CHECK(Yoojin): Expectation, Mean
                #CHECK(Yoojin): Only two loss term in evaluation
                encoded_loss = criterion(encoded_trump, trump_val_x) + criterion(encoded_cage, cage_val_x) # Eq13, MSE Loss
                encoded_loss /= 2 

                image_adv_logits_loss = (
                        criterion(logits_original_trump_val, logits_trump_endf_val) + \
                        criterion(logits_original_cage_val, logits_cage_endf_val)
                )
                image_adv_logits_loss /= 2 

                image_loss = 0.9 * encoded_loss + 0.1 * image_adv_logits_loss # ori=0.9, adv=0.1
            #    image_loss *= args.alpha_val


                if epoch >= d_iter:
                    # Pred from encoded deepfake image
                    encoded_trump_df_pred_message = model.decode(encoded_trump_df)
                    encoded_cage_df_pred_message = model.decode(encoded_cage_df)

                    # Pred from encoded original image
                    encoded_trump_pred_message = model.decode(encoded_trump)
                    encoded_cage_pred_message = model.decode(encoded_cage)

                    # compute message_loss
                    message_loss = message_criterion(encoded_trump_df_pred_message, trump_message) + \
                                   message_criterion(encoded_cage_df_pred_message, cage_message) + \
                                   message_criterion(encoded_trump_pred_message, trump_message) + \
                                   message_criterion(encoded_cage_pred_message, cage_message)
                    message_loss /= 4 

                    val_df_message_correct += ((encoded_trump_df_pred_message > 0.5) == trump_message).sum().item() + ((encoded_cage_df_pred_message > 0.5) == cage_message).sum().item()
                    val_message_correct += ((encoded_trump_pred_message > 0.5) == trump_message).sum().item() + ((encoded_cage_pred_message > 0.5) == cage_message).sum().item()
                    
                    #Modified(Yoojin): No beta multiplied
                    val_message_loss += message_loss.item()
                
                val_image_loss += encoded_loss.item()
                val_image_df_loss += image_adv_logits_loss.item()
                val_size += trump_val_x.shape[0] + cage_val_x.shape[0]
            
            # val_image_loss /= val_size
            # val_image_df_loss /= val_size
            # val_message_loss /= val_size

            val_message_acc = val_message_correct / (val_size * message_size)
            val_df_message_acc = val_df_message_correct / (val_size * message_size)

            #CHECK
            val_loss = alpha*(val_image_loss * 0.9 + val_image_df_loss * 0.1) + beta*val_message_loss

            val_ed_loss.append(val_image_loss)
            val_adv_loss.append(val_image_df_loss)
            val_loss_plot.append(val_message_loss)

            val_acc_plot.append(val_message_acc)
            val_df_acc_plot.append(val_df_message_acc)

            f_print = open(os.path.join(save_path, "train_loss_log.txt"), "a")
            print(f"Til: {train_image_loss}, Tdl: {train_image_df_loss}, Tga: {train_gan_loss}", # Tssim: {train_ssim_loss}", #Modified(Yoojin)
                  f"Vil: {val_image_loss}, Vdl:{val_image_df_loss}",
                  f"Tml: {train_message_loss}, Vml: {val_message_loss}",
                  f"Ta: {train_message_acc}, TFa:{train_df_message_acc}",
                  f"Va: {val_message_acc}, VFa: {val_df_message_acc}", file=f_print)
            f_print.close()
            
            path = os.path.join(save_path, name) 

            if epoch > d_iter:
                if epoch == d_iter + 1:
                    f_print = open(os.path.join(save_path, "train_loss_log.txt"), "a")
                    print(f">> Starts training decoder\n", file=f_print)
                    f_print.close()    
                if epoch % 5 == 0:
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    torch.save({
                        "encoder": model.encoder.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "epoch": epoch
                    }, os.path.join(path, f'ckpt_e{epoch}.pt'))
                    print(f'Model saved at epoch {epoch + 1}/{epochs}')

                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    torch.save({
                        "encoder": model.encoder.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "epoch": epoch
                    }, os.path.join(path, ckpt_best))
                    print(f'(img+msg)model saved at epoch {epoch}')

            if min_val_image_loss > val_image_loss and epoch > d_iter:               
                min_val_image_loss = val_image_loss
                if not os.path.isdir(path):
                    os.makedirs(path)
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": epoch
                }, os.path.join(path,ckpt_img_best))
                print(f'(img)model saved at epoch {epoch}')

            if epoch == epochs - 1: # Last epochs
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": epoch
                }, os.path.join(path, ckpt_final))
            

        # End of Evaluation
        if epoch % 5 == 0:
            plt.plot(np.arange(len(val_loss_plot)), val_loss_plot, label="loss")
            plt.plot(np.arange(len(val_acc_plot)), val_acc_plot, label="valid acc")
            plt.plot(np.arange(len(val_df_acc_plot)), val_df_acc_plot, label="valid df acc")
            plt.legend()
            plt.savefig(os.path.join(save_path,'loss', "val_loss_acc.png"))
            plt.close()

            plt.plot(np.arange(len(train_ed_loss)), train_ed_loss, label="train image loss")
            plt.plot(np.arange(len(train_adv_loss)), train_adv_loss, label="train adv loss")
            plt.plot(np.arange(len(val_ed_loss)), val_ed_loss, label="val image loss")
            plt.plot(np.arange(len(val_adv_loss)), val_adv_loss, label="val adv loss")
            plt.savefig(os.path.join(save_path,'loss', "trval_loss_img.png"))
            plt.legend()
            plt.close() 

            plt.plot(np.arange(len(train_ed_loss)), train_ed_loss, label="image loss")
            plt.plot(np.arange(len(train_adv_loss)), train_adv_loss, label="adv loss")
            # plt.plot(np.arange(len(train_ssim_loss_plot)), train_ssim_loss_plot, label="ssim loss")
            plt.plot(np.arange(len(train_gan_loss_plot)), train_gan_loss_plot, label="gan loss")
            plt.plot(np.arange(len(train_loss_plot)), train_loss_plot, label="message loss") 
            plt.legend()
            plt.savefig(os.path.join(save_path,'loss', "train_loss_image.png"))
            plt.close()
    # End of for loop (Epochs)        
    
#NOTE(Yoojin): Deleted `test` function in `train.py`
DEVICE_IDS = [0, 4]
def main():
    set_seed(0) 
    args = default_argument_parser() 

    config_path = args.config_path 
    save_path = args.save_path 
    
    train_config = load_yaml(config_path)['train']
    trump_path = train_config['trump_path'] 
    cage_path = train_config['cage_path'] 
    device = torch.device(f'cuda:{DEVICE_IDS[0]}' if torch.cuda.is_available() else 'cpu') 

    os.makedirs(os.path.join(save_path, 'loss'), exist_ok=True)
    shutil.copyfile(args.config_path, os.path.join(save_path, 'config.yml'))
    print(f'[DEBUG] Trained Model will be saved at {save_path}') 

    print("[INFO] Split dataset...")
    trump_train_dataset, trump_val_dataset, _ = split_dataset(trump_path, train_transform=transform_train, valid_transform=transform_val, test_transform=transform_test, val_ratio=0.1, test_ratio=0.1)
    cage_train_dataset, cage_val_dataset, _ = split_dataset(cage_path, train_transform=transform_train,valid_transform=transform_val, test_transform=transform_test, val_ratio=0.1, test_ratio=0.1)

    print("[INFO] Load dataset...")
    trump_train_loader = DataLoader(trump_train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    trump_val_loader = DataLoader(trump_val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    cage_train_loader = DataLoader(cage_train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    cage_val_loader = DataLoader(cage_val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    assert len(trump_train_loader) == len(cage_train_loader), "The size of two dataset should be same"

    criterion = nn.MSELoss(reduction='mean')
    message_criterion = nn.BCELoss(reduction='mean') 
    adversarial_loss = nn.BCELoss(reduce='mean') 
    epochs = train_config['epoch']
    message_size = train_config['message_size']
    ckpt_names = get_ckpt_name(train_config) 

    model = DualDefense(message_size= message_size, in_channels=3) #,device=device) 
    if torch.cuda.device_count() > 1:
        model = engine.utils.DataParallel(model, device_ids=DEVICE_IDS)
    model.to(device)

    optimizer = optim.Adam(
            params=list(model.encoder.parameters())+ list(model.decoder.parameters()), 
            lr=train_config['lr'], 
            weight_decay=1e-5
        )
    optimizer_discriminator = optim.Adam(
        params=model.adv_model.parameters(),
        lr=train_config['lr'],
        weight_decay=1e-5
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=train_config['T_max'], 
        eta_min=1e-8
    )
    configs = (epochs, message_size, train_config['start_decode'], train_config['clip'], save_path, train_config['name'], train_config['lambda_val'], train_config['alpha_val']) 

    print("[INFO] Starts training...")
    train(configs,
          model, 
          trump_train_loader, cage_train_loader,trump_val_loader, cage_val_loader,
          optimizer, optimizer_discriminator,
          lr_scheduler, criterion, message_criterion,
          adversarial_loss, 
          ckpt_names, device 
          )

if __name__ == "__main__":
    main()