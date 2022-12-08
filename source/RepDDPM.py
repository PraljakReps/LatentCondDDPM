
import torch
import torch.nn as nn
import torch.nn.functional as F
from source.module import DoubleConv, Down, Up, SelfAttention, one_param
import numpy as np 



class Rep_Down(nn.Module):
    """
    This layer is different than the Down layer because it
    does not leverage time-conditional embeddings.
    """

    def __init__(self, in_channels, out_channels, emb_dim = 256):
        
        super().__init__()

        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    residual = True
                ),
                DoubleConv(
                    in_channels = in_channels,
                    out_channels = out_channels
                )
        )

        self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_dim,
                    out_channels,
                ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Rep_Up(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)
        self.conv = nn.Sequential(
                DoubleConv(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    residual = True
                ),
                DoubleConv(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    mid_channels = in_channels // 2,
                ),
        )
        
        self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_dim,
                    out_channels
                ),
        )

    def forward(self, x, t):

        x = self.up(x)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb


class Rep_AE(nn.Module):

    def __init__(self, c_in, c_out, z_dim = 100, time_dim = 256):
        super().__init__()


        self.emb_dim = time_dim
        
        self.inc = DoubleConv(c_in, 64)
        
        # downsampling encoder:

        self.down1 = Rep_Down(64, 128, emb_dim = self.emb_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Rep_Down(128, 256, emb_dim = self.emb_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Rep_Down(256, 256, emb_dim = self.emb_dim)
        self.sa3 = SelfAttention(256)

        self.output_size = 256 * 4 * 4
        self.z_dim = z_dim
       
        self.q_z_mean = nn.Linear(self.output_size, self.z_dim)
        self.q_z_var = nn.Sequential(
                nn.Linear(self.output_size, self.z_dim),
                nn.Softplus()
        )
 
        self.bn_size = 256
        self.bn_width = 4
        self.bn_height = 4
        self.z_up_linear = nn.Linear(self.z_dim, self.bn_size * self.bn_width * self.bn_height)

        self.up1 = Rep_Up(256, 256, emb_dim = self.emb_dim)
        self.sa1_up = SelfAttention(256)
        self.up2 = Rep_Up(256, 128, emb_dim = self.emb_dim)
        self.sa2_up = SelfAttention(128)
        self.up3 = Rep_Up(128, 64, emb_dim = self.emb_dim)
        self.sa3_up = SelfAttention(64)

        self.outc = nn.Conv2d(64, c_out, kernel_size = 1)
        

    def encoder_forward(self,x, t):

        h = self.inc(x)
        h = self.sa1(self.down1(h, t))
        h = self.sa2(self.down2(h, t))
        h = self.sa3(self.down3(h, t))
        
        # flatten the last dimensions
        h = h.reshape(h.shape[0], -1)
        
        return h

    def decoder_forward(self, z, t):
        

        h = self.up1(z, t)
        h = self.sa1_up(h)

        h = self.up2(h, t)
        h = self.sa2_up(h)

        h = self.up3(h, t)
        h = self.sa3_up(h)

        out = self.outc(h)

        return out
        

    @staticmethod
    def compute_kernel(x, y):
        
        # size of the mini-batches
        x_size, y_size = x.shape[0], y.shape[0]

        # dimension based on z-axis
        dim = x.shape[1] # can also be considered as a hyperparameter

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)

        x_core = x.expand(x_size, y_size, dim)
        y_core = y.expand(x_size, y_size, dim)

        return torch.exp(-(x_core - y_core).pow(2).mean(2)/dim)


    def mmd_loss(self, x, y):
        """
        function description: compute max-mean discrepancy
        arg:
            x --> random distribution z~p(x)
            y --> embedding distribution z'~q(z)
        return:
            MMD_loss --> max-mean discrepancy loss between the sampled noise 
            and embedded distribution
        """
        
        x_kernel = Rep_AE.compute_kernel(x,x)
        y_kernel = Rep_AE.compute_kernel(y,y)
        xy_kernel = Rep_AE.compute_kernel(x,y)
        
        return x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()

    def reparam_trick(self, z_mean, z_var):
    
        return z_mean + z_var * torch.randn_like(z_var)

    def kl_loss(self, z_mu, z_var):

        # posterior kl-divergence loss (assuming normal prior):
        loss_kld = torch.mean(-0.5 * torch.sum(1 + z_var.log() - z_mu ** 2 - z_var, dim = 1), dim = 0)

        return loss_kld

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device = one_param(self).device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim = -1)

        return pos_enc

    def latent_upscale(self, z):

        z_up = self.z_up_linear(z)
        z_up = z_up.reshape(z_up.shape[0], self.bn_size, self.bn_height, self.bn_width)
        return z_up

    def latent_encoder(self, h):
        z_mean = self.q_z_mean(h)
        z_var = self.q_z_var(h)
        z = self.reparam_trick(z_mean, z_var)
        z_up = self.latent_upscale(z)

        return z_up, z, z_mean, z_var

    def forward(self, x , t):

        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.emb_dim)
        h = self.encoder_forward(x, t)

        z_up, z, z_mean, z_var = self.latent_encoder(h)

        out = self.decoder_forward(z_up, t)

        return out, z, z_mean, z_var


class RepCond_AttUnet(nn.Module):

    def __init__(self, c_in = 3, c_out = 3, time_dim = 256):
        super().__init__()

        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        # bottleneck layer 
        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)
        
    
        # decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size = 1)


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device = one_param(self).device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim = -1)

        return pos_enc


    def unet_forward(self, x, t):

        x1 = self.inc(x)

        # downscale
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 - self.sa3(x4)

        # bottleneck
        bot_x1 = self.bot1(x4)
        bot_x2 = self.bot2(bot_x1)

        # upscale
        x_up = self.up1(bot_x2, x3, t)
        x_up = self.sa4(x_up)
        x_up = self.up2(x_up, x2, t)
        x_up = self.sa5(x_up)
        x_up = self.up3(x_up, x1, t)
        x_up = self.sa6(x_up)

        # output (reconstruction)
        output = self.outc(x_up)
        return output

    def forward(self, x, x_z, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        
        x_cond = torch.cat((x_z, x), dim = 1)

        return self.unet_forward(x_cond, t)         


class RepDDPM(nn.Module):


    def __init__(self, Unet, latent_AE, diffusion, alpha = 0.99, beta = 10):
        super().__init__()
        self.Unet = Unet
        self.latent_AE = latent_AE
        self.diffusion = diffusion

        self.alpha = alpha
        self.beta = beta


    def compute_loss(self, true_noise, pred_noise, z_true, z_infer, z_mu, z_var, gamma = 1e-1):
       
        # DDPM loss, where we take the MSE between the added noise and predicted noise 
        mse = nn.MSELoss(reduction = 'none')
        loss_ddpm = torch.mean(
                torch.sum(
                    mse(true_noise, pred_noise), dim = (-2, -1)
                )
        )

        # kullback-liebler divergence between q(z|x) and p(z)
        loss_kl = self.latent_AE.kl_loss(z_mu, z_var)

        # max mean discrepancy loss between q(z) and p(z)
        loss_mmd = self.latent_AE.mmd_loss(z_true, z_infer)

        loss = loss_ddpm + gamma * ( (1 - self.alpha) * loss_kl + (self.alpha + self.beta - 1) * loss_mmd )

        return loss, loss_ddpm, loss_kl, loss_mmd

    def sample_randomly(self, z, t ):
        t = t.unsqueeze(-1)
        t = self.latent_AE.pos_encoding(t, self.latent_AE.emb_dim)

        z_up = self.latent_AE.latent_upscale(z)
        out = self.latent_AE.decoder_forward(z_up, t)
        return out

    def forward(self, x, t, x0):

        x_z, z, z_mean, z_var = self.latent_AE(x0, t)
        
        E = self.Unet(x, x_z, t)       

        return E, z, z_mean, z_var
