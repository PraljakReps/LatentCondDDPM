import torch
import torch.nn as nn
import torch.nn.functional as F
from source.module import DoubleConv, Down, Up, SelfAttention, one_param
import numpy as np


class TM_hidden_layer(nn.Module):

    def __init__(self, in_width, out_width):

        super().__init__()

        self.in_width = in_width
        self.out_width = out_width
        self.hidden_layer = nn.Sequential(
                nn.Linear(self.in_width, self.out_width),
                nn.LayerNorm(self.out_width),
                nn.SiLU()
        )

    def forward(self, x):
        return self.hidden_layer(x)


class top_model(nn.Module):

    def __init__(self, num_layers, hidden_width, z_dim, num_classes):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.z_dim = z_dim
        self.num_classes = num_classes

        

        self.top_model_layers = nn.ModuleList()

        for ii in range(num_layers):

            if ii == 0:

                self.top_model_layers.append(TM_hidden_layer(self.z_dim, self.hidden_width))

            else:
                
                self.top_model_layers.append(TM_hidden_layer(self.hidden_width, self.hidden_width))


        self.output_layer = nn.Linear(self.hidden_width, self.num_classes)
        self.softmax = nn.Softmax(dim = -1)


    def forward(self, z):

        for layer in self.top_model_layers:
            z = layer(z)
        
        h = self.output_layer(z)
        return h


class SemiSup_DDPM(nn.Module):

    def __init__(self, Unet, latent_AE, top_model, diffusion, xi = 1, alpha = 0.99, beta = 10, gamma = 1):
        super().__init__()

        self.Unet = Unet
        self.latent_AE = latent_AE
        self.top_model = top_model
        self.diffusion = diffusion

        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_loss(
            self,
            true_noise,
            pred_noise,
            z_true,
            z_infer,
            z_mu,
            z_var,
            y_true,
            y_pred
        ):

        # DDPM loss
        mse = nn.MSELoss(reduction = 'none')
        loss_ddpm = torch.mean(
                torch.sum(
                    mse(true_noise, pred_noise), dim = (-2, -1)
                    )
        )

        # KL divergence 
        loss_kl = self.latent_AE.kl_loss(z_mu, z_var)

        # max mean discrepancy loss
        loss_mmd = self.latent_AE.mmd_loss(z_true, z_infer)

        # discriminative loss:
        ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        y_true = torch.argmax(y_true, dim = -1).long()
        loss_disc = ce_loss(y_pred, y_true)
        loss_disc = torch.mean(loss_disc, dim = -1)
        
        
        loss = self.xi * loss_ddpm + (1-self.alpha)*loss_kl + (self.alpha + self.beta - 1)*loss_mmd + self.gamma*loss_disc
        return loss, loss_ddpm, loss_kl, loss_mmd, loss_disc

    
    def sample_randomly(self, z, t):

        t = t.unsqueeze(-1)
        t = self.latent_AE.pos_encoding(t, self.latent_AE.emb_dim)

        z_up = self.latent_AE.latent_upscale(z)
        out = self.latent_AE.decoder_forward(z_up, t)
        return out

    def forward(self, x, t, x0):

        x_z, z, z_mean, z_var = self.latent_AE(x0, t)
        y_pred = self.top_model(z) # discriminative top model
        E = self.Unet(x, x_z, t) # DDPM: predict noise

        return E, z, z_mean, z_var, y_pred


