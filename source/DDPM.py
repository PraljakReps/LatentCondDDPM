import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from source.module import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
import math





class Diffusion:

    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 256, schedule = 'linear', device = 'cuda'):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule = schedule

        self.beta = self.prepare_noise_schedule(schedule = self.schedule).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)


    def cosine_beta_schedule(self, s = 0.008):
        """
        cosine schedule proprosed in https://arxiv.org/abs/2102.09672
        """
        timesteps = self.noise_steps

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    
    def linear_beta_schedule(self,):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.noise_steps)



    def prepare_noise_schedule(self, schedule = 'linear'):

        if schedule == 'cosine':
            return self.cosine_beta_schedule()
        
        else:
            return self.linear_beta_schedule()


    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        E = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * E, E

    def sample_timesteps(self, n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))




