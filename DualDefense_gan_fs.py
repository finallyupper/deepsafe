import torch
import torch.nn as nn
from adv import Adversary_Init
from encoder_fs import ResNetUNet
from decoder_fs import Decoder
from faceswap_pytorch.models import Autoencoder_df

CHECKPOINTS = {
    'trump_cage' : '/data/yoojinoh/def/241116_faceswap_160/checkpoint/autoencoder_160.t7', 
    'winter_karina': '/content/drive/MyDrive/prometheus/deepsafe/data/ae_win_kar_160.t7'
}

class DualDefense(nn.Module):
    def __init__(self, message_size, in_channels, device):
        super().__init__()
        self.encoder = ResNetUNet(message_size)
        self.df_model = Autoencoder_df()
        self.decoder = Decoder(message_size)
        self.adv_model = Adversary_Init(in_channels) 
#        checkpoint = torch.load(CHECKPOINTS['winter_karina'])
        checkpoint = torch.load(CHECKPOINTS['winter_karina'], map_location=device)


        self.df_model.load_state_dict(checkpoint['state']) 

        if device:
            self.encoder = self.encoder.to(device)
            self.df_model = self.df_model.to(device)
            self.decoder = self.decoder.to(device)
            self.adv_model = self.adv_model.to(device)

    def encode(self, x, message):
        return self.encoder(x, message)

    def deepfake1(self, x, type):
        return self.df_model(x, type)

    def adv(self, x):
        return self.adv_model(x)

    def decode(self, x):
        return self.decoder(x)
