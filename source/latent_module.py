
import torch
import torch.nn as nn
import torch.nn.functional as F
from source.vanilla_module import DoubleConv, Down, Up, SelfAttention, one_param, Att_enc_block, Att_dec_block
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

    """
    Same function as Up function used in the DDPM U-net; however, this layer doesn't take skip connections from the encoder.
    """

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


class Rep_dec_block(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            emb_dim
        ):

        super().__init__()
        self.Up_layer = Rep_Up(in_channels, out_channels, emb_dim)
        self.SelfAtt = SelfAttention(out_channels)

    def forward(self, x, t):
        h = self.Up_layer(x, t)
        return self.SelfAtt(h)



class Rep_AE(nn.Module):

    def __init__(
            self,
            img_size,
            c_in,
            c_out,
            first_num_channel: int = 64,
            z_dim: int = 100,
            time_dim: int = 256,
            num_layers: int = 3,
            bn_layers: int = 2
        ):
        super().__init__()

        self.img_size = img_size
        self.emb_dim = time_dim
        self.z_dim = z_dim
        self.first_num_channel = first_num_channel
        self.num_layers = num_layers
        self.bn_layers = bn_layers



        self.inc = DoubleConv(c_in, first_num_channel)
        
        self.encoder = nn.ModuleList()

        in_channels = self.first_num_channel
        # downsampling encoder:
        
        for ii in range(self.num_layers):
            # define output channel as twice the size
            out_channels = in_channels * 2

            # encoder module
            self.encoder.append(
                    Att_enc_block(
                        in_channels = in_channels, out_channels = out_channels
                        )
            )

            # increase the input size
            in_channels = out_channels


        # latent inference
        self.encoder_output_size = self.img_size // 2**(self.num_layers)
        self.output_size = out_channels * self.encoder_output_size ** 2 
    
        self.q_z_mean = nn.Linear(self.output_size, self.z_dim)
        self.q_z_var = nn.Sequential(
                nn.Linear(self.output_size, self.z_dim),
                nn.Softplus()
        )
        # define variables for bottleneck tensor parameters
        self.bn_height = self.encoder_output_size
        self.bn_width = self.encoder_output_size
        self.bn_size = out_channels
        
        self.z_up_linear = nn.Linear(self.z_dim, self.output_size)

        
        in_channels = out_channels
        
        # define decoder module
        self.decoder = nn.ModuleList()
        for ii in range(self.num_layers):
            
            out_channels = in_channels // 2
            self.decoder.append(Rep_dec_block(
                in_channels = in_channels,
                out_channels = out_channels,
                emb_dim = self.emb_dim
                )
            )
        
            in_channels = out_channels

        self.outc = nn.Conv2d(out_channels, c_out, kernel_size = 1)
        

    def encoder_forward(self,x, t):

        h = self.inc(x)

        for layer in self.encoder:
            h = layer(h, t)

        # flatten the last dimensions
        h = h.reshape(h.shape[0], -1)
        
        return h

    def decoder_forward(self, z, t):
       

        for layer in self.decoder:
            z = layer(z,t)

        out = self.outc(z)

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


class LatentCond_Unet(nn.Module):

    def __init__(
            self,
            c_in: int = 3,
            c_out: int = 3,
            first_num_channel: int = 64,
            time_dim: int = 256,
            num_layers: int = 3,
            bn_layers: int = 2):
        super().__init__()
    
        self.first_num_channel = first_num_channel
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.bn_layers = bn_layers
        

        self.inc = DoubleConv(c_in, self.first_num_channel)
        
        self.encoder = nn.ModuleList()

        in_channels = self.first_num_channel
        self.en_in_channels, self.en_out_channels = [], []

        for ii in range(self.num_layers):

            # define output channel as twice the size
            out_channels = in_channels * 2
            
            # log the input and output channels for each encoder hidden block
            self.en_in_channels.append(in_channels)
            self.en_out_channels.append(out_channels)

            # encoder module 
            self.encoder.append(Att_enc_block(
                in_channels = in_channels, out_channels = out_channels
                )
            )
            

            # increase the input size
            in_channels = out_channels        

        # Unet's bottleneck
        
        self.bn_nn = nn.ModuleList()
        for ii in range(self.bn_layers):
            self.bn_nn.append(DoubleConv(out_channels, out_channels))

        # decoder module:
        self.decoder = nn.ModuleList()

        for ii, (in_c, out_c) in enumerate(zip( reversed(self.en_in_channels), reversed(self.en_out_channels) )):

            in_channels = in_c + out_channels
            out_channels = in_c
            # decoder module
            self.decoder.append(Att_dec_block(
                in_channels = in_channels,
                out_channels = out_channels
                )               
            )

        self.outc = nn.Conv2d(out_channels, c_out, kernel_size = 1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device = one_param(self).device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2)* inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim = -1)
        return pos_enc

    def unet_forward(self, x, t):

        h = self.inc(x)
        

        skip_connections = []

        skip_connections.append(h)

        for encoder_layer in self.encoder:
            
            h = encoder_layer(h,t)

            skip_connections.append(h)

        for bn_layer in self.bn_nn:
            h = bn_layer(h)


        # remove last appended skip before entering for loop ...
        skip_connections.pop()
        for decoder_layer, h_skip in zip(self.decoder, reversed(skip_connections)):
            h = decoder_layer(h, h_skip, t)

        # output (predict error):
        output = self.outc(h)

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
