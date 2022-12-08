
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import argparse
import sys
import pandas as pd

from source.latent_module import Rep_AE, LatentCond_Unet, RepDDPM
from source.DDPM import Diffusion
import source.helper_func as hf
import logging
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision import transforms as T
import math






def set_seed(SEED: int):

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

def prepare_data(
        task: str = 'mnist',
        batch_size: int = 32
    ):


    if task == 'mnist':
        transforms = T.Compose(
                [
                    T.Resize([32,32]),
                    T.ToTensor(),
                    T.Normalize((0.5), (0.5))
                ]
        )


        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    else:
        transforms = T.Compose(
                [
                    T.Resize([32,32]),
                    T.ToTensor(),
                    T.Normalize((0.5),(0.5))
                ]
        )
        train_transforms = T.Compose(
                [
                    T.Resize([32,32]),
                    T.ToTensor(),
                    T.Normalize((0.5),(0.5)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5)
                ]
        )
            
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
        
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)


    return train_dataloader, test_dataloader


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-S', '--SEED', default = 42, help = 'flag: random seed', type = int)
    parser.add_argument('-bs', '--batch_size', default = 32, help = 'flag: batch size', type = int)
    parser.add_argument('-lr', '--learning_rate', default = 1e-3, help = 'flag: learning rate', type = float)
    parser.add_argument('-e', '--epochs', default = 10, help = 'flag: training epochs', type = int)
    parser.add_argument('-t', '--tasks', default = 'mnist', help = 'flag: pick dataset', type = str)
   

    # model hyperparameter
    parser.add_argument('-ig', '--img_size', default = 32, help = 'flag: image size', type = int)
    parser.add_argument('-cin', '--c_in', default = 1, help = 'flag: input channel to the Unet model', type = int)
    parser.add_argument('-cout', '--c_out', default = 1, help = 'flag: output channel to the Unet model', type = int)
    parser.add_argument('-fnc', '--first_num_channel', default = 64, help = 'flag: number of conv channels for the first layer', type = int)
    parser.add_argument('-td', '--time_dim', default = 256, help = 'flag: embedding dimension for the time positions', type = int)
    parser.add_argument('-nl', '--num_layers', default = 3, help = 'flag: number of layers', type = int)
    parser.add_argument('-bn', '--bn_layers', default = 2, help = 'flag: number of Unets bottleneck layers', type = int)
    parser.add_argument('-rl', '--rep_learning', default = False, help = 'flag: choose to use latent conditional info or not.', type = bool)
    parser.add_argument('-zd', '--z_dim', default = 5, help = 'flag: latent space dimension', type = int)

    # learning prefactors
    parser.add_argument('-al', '--alpha', default = 0.99, help = 'flag: mutual information (x,z) prefactor weight', type = float)
    parser.add_argument('-be', '--beta', default = 10, help = 'flag: MMD prefactor weight', type = float)
    

    # filepaths
    parser.add_argument('-fop', '--fig_output_path', default = './output/vanillaDDPM/mnist_samples_', help = 'flag: filenames for the mnist files', type = str)
    parser.add_argument('-omp', '--output_model_path', default = './output/vanillaDDPM/model_pretrained_weights.pth', help = 'flag: model weights', type = str)
    parser.add_argument('-ocp', '--output_csv_path', default = './output/vanillaDDPM/model_train_history.csv', help = 'flag: model training/testing loss history', type = str)

    args = parser.parse_args()

    return args

@torch.no_grad()
def sample(
        diffusion: any,
        model: any,
        n: int = 20
    ):
    
    model.eval()
    
        
    x_sample = torch.randn((n, 1, diffusion.img_size, diffusion.img_size)).to(diffusion.device)
    z_0 = torch.randn((n, model.latent_AE.z_dim)).to(DEVICE)

    for ii in tqdm(reversed(range(1, diffusion.noise_steps)), position = 0):
            
        t = (torch.ones(n) * ii).long().to(diffusion.device)
            
        # decode latents
        x_z = model.sample_randomly(z_0, t)

        # predict noise
        predicted_noise = model.Unet(x_sample, x_z, t)

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
            
        if ii > 1:
            noise = torch.randn_like(x_sample)
        else:
            noise = torch.zeros_like(x_sample)
                
                
        alpha_prefactor = (1-alpha)/ (torch.sqrt(1-alpha_hat))

        x_sample = 1 / torch.sqrt(alpha) * (x_sample - alpha_prefactor*predicted_noise) + torch.sqrt(beta) * noise
            
    model.train()
    x_sample = (x_sample.clamp(-1, 1) + 1)/ 2
    return x_sample


def plot_samples(
        fig_output_path: str,
        X_sample: torch.FloatTensor,
        epoch: int
    ):

    fig, axes = plt.subplots(5,4, dpi = 300, sharex = True, sharey = True)

    fig.subplots_adjust(hspace = 0, wspace = 0)

    idx = 0
    for ii in range(5):

        for jj in range(4):

            axes[ii,jj].imshow(
                    X_sample[idx, 0].cpu()
            )
            axes[ii,jj].axis('off')

            axes[ii,jj].set_xticklabels([])
            axes[ii,jj].set_yticklabels([])
            idx+=1
    plt.tight_layout()
    plt.savefig(f'{fig_output_path}_epoch={epoch}.png', dpi = 300)
    

@torch.no_grad()
def end_epoch_validation(
        DEVICE: str,
        model: any,
        diffusion: any,
        test_dataloader: any
    ):
    
    model.eval()
    
    epoch_loss, epoch_mse_loss, epoch_kld_loss, epoch_mmd_loss = [], [], [], []
    for iteration, valid_batch in enumerate(test_dataloader):
        
        # validation batch using the testing set
        valid_batch = Variable(valid_batch[0].to(DEVICE))
        
        # timesteps for each batch
        t = diffusion.sample_timesteps(valid_batch.shape[0]).to(DEVICE)    
            
        # corrupt training batch
        x_t, noise = diffusion.noise_images(valid_batch, t)

        # predict noise 
        predicted_noise, z, z_mean, z_var = model(x_t, t, valid_batch)
        
        # sample from some noise:
        # normal dist:
        if DEVICE == 'cuda':
            z_true_samples = torch.randn_like(z).to(DEVICE)
        
        else:
            z_true_samples = torch.randn_like(z)
            
    
        # loss
        loss, loss_ddpm, loss_kld, loss_mmd = model.compute_loss(
                                        true_noise = noise,
                                        pred_noise = predicted_noise,
                                        z_true = z_true_samples,
                                        z_infer = z,
                                        z_mu = z_mean,
                                        z_var = z_var,
                                        gamma = 1
        )

        epoch_loss.append(loss.item())
        epoch_mse_loss.append(loss_ddpm.item())
        epoch_kld_loss.append(loss_kld.item())
        epoch_mmd_loss.append(loss_mmd.item())
        
    model.train()
    
    print(f'Validation | loss: {np.mean(epoch_loss)} | Mean squared error: {np.mean(epoch_mse_loss)} | KLD loss: {np.mean(epoch_kld_loss)} | MMD loss: {np.mean(epoch_mmd_loss)}')

    return np.mean(epoch_loss), np.mean(epoch_mse_loss), np.mean(epoch_kld_loss), np.mean(epoch_mmd_loss)
       

def train_model(
    train_dataloader: any,
    test_dataloader: any,
    args: any,
    DEVICE: str = 'cuda'
    ) -> (any, dict):
    
    # model hyperparameters
    img_size = args.img_size
    c_in = args.c_in
    c_out = args.c_out
    first_num_channel = args.first_num_channel
    time_dim = args.time_dim
    num_layers = args.num_layers
    bn_layers = args.bn_layers
    z_dim = args.z_dim

    # loss prefactors
    alpha = args.alpha
    beta = args.beta

    # configure model
    rep_AE = Rep_AE(
            img_size = img_size,
            c_in = c_in,
            c_out = c_out,
            first_num_channel = first_num_channel,
            z_dim = z_dim,
            time_dim = time_dim,
            num_layers = num_layers,
            bn_layers = bn_layers
    ).to(DEVICE)

    Unet = LatentCond_Unet(
            c_in = c_in + c_in,
            c_out = c_out,
            first_num_channel = first_num_channel,
            time_dim = time_dim,
            num_layers = num_layers,
            bn_layers = bn_layers
    ).to(DEVICE)

    
    diffusion = Diffusion(
            img_size = img_size,
            device =  DEVICE,
            schedule = 'linear'
    )

    model = RepDDPM(
            Unet = Unet,
            latent_AE = rep_AE,
            diffusion = diffusion,
            alpha = alpha,
            beta = beta
    ).to(DEVICE)


    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    mse = nn.MSELoss(reduction = 'none')
    model.train()
    
    history_dict = {
        'epoch': list(),
        'train_loss': list(),
        'test_loss': list(),
        'train_mse_loss': list(),
        'test_mse_loss': list(),
        'train_kld_loss': list(),
        'test_kld_loss': list(),
        'train_mmd_loss': list(),
        'test_mmd_loss': list()
    }

    # train model
    for epoch in range(args.epochs):
        
        epoch_loss, epoch_mse_loss, epoch_kld_loss, epoch_mmd_loss = [], [], [], []
        for iteration, train_batch in tqdm(enumerate(train_dataloader)):

            # training batch
            train_batch = Variable(train_batch[0].to(DEVICE))

            # timesteps for each batch
            t = diffusion.sample_timesteps(train_batch.shape[0]).to(DEVICE)

            # corrupt training batch
            x_t, noise = diffusion.noise_images(train_batch, t)

            # predicted noise
            predicted_noise, z, z_mean, z_var = model(x = x_t, t = t, x0 = train_batch)
            
            # sample from some noise
            if DEVICE == 'cuda':
                z_true_samples = Variable(torch.randn_like(z)).to(DEVICE)
            else:
                z_true_samples = torch.randn_like(z)

            # loss
            loss, loss_ddpm, loss_kld, loss_mmd = model.compute_loss(
                                                    true_noise = noise,
                                                    pred_noise = predicted_noise,
                                                    z_true = z_true_samples,
                                                    z_infer = z,
                                                    z_mu = z_mean,
                                                    z_var = z_var,
                                                    gamma = 1
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_mse_loss.append(loss_ddpm.item())
            epoch_kld_loss.append(loss_kld.item())
            epoch_mmd_loss.append(loss_mmd.item())
           

        train_loss = np.mean(epoch_loss)
        train_mse_loss = np.mean(epoch_mse_loss)
        train_kld_loss = np.mean(epoch_kld_loss)
        train_mmd_loss = np.mean(epoch_mmd_loss)

        print(f'Training | loss: {train_loss} ' +
        f'| Mean squared error: {train_mse_loss} ' +
        f'| KLD loss: {train_kld_loss} ' +
        f'| MMD loss: {train_mmd_loss}')
        
        history_dict['epoch'].append(epoch)
        history_dict['train_loss'].append(train_loss)
        history_dict['train_mse_loss'].append(train_mse_loss)
        history_dict['train_kld_loss'].append(train_kld_loss)
        history_dict['train_mmd_loss'].append(train_mmd_loss)
        
        test_loss, test_mse_loss, test_kld_loss, test_mmd_loss = end_epoch_validation(
            DEVICE = DEVICE,
            model = model,
            diffusion = diffusion,
            test_dataloader = test_dataloader
        )
    
        history_dict['test_loss'].append(test_loss)
        history_dict['test_mse_loss'].append(test_mse_loss)
        history_dict['test_kld_loss'].append(test_kld_loss)
        history_dict['test_mmd_loss'].append(test_mmd_loss)
        
        # generate samples at the end of epoch
        X_sample = hf.generate_sample(
                args = args,
                task = args.tasks,
                diffusion = diffusion,
                model = model,
                n = 20,
                DEVICE = DEVICE
        )
        
        # save generate samples
        hf.plot_random_samples(
            args = args,
            sample_images = X_sample,
        )
        plt.savefig(f'{args.fig_output_path}_epoch={epoch}.png', dpi = 300)
    
    return model, history_dict


if __name__ == '__main__':

    # get arguments
    args = get_args()
    
    SEED = args.SEED # random seed
    batch_size = args.batch_size # batch size
    output_model_path = args.output_model_path # model output path
    output_csv_path = args.output_csv_path # train/test loss history path
    task = args.tasks # pick dataset

    # set seed for reproducibility...
    set_seed(SEED = SEED)

    # check GPU
    if torch.cuda.is_available():
        print('GPU available')

    else:
        print('Please enable GPU and rerun script')
        quit()


    USE_CUDA = True
    DEVICE = 'cuda' if USE_CUDA else 'cpu'


    # prepare data
    train_dataloader, test_dataloader = prepare_data(
                                                task = task,
                                                batch_size=batch_size
    )


    # train model
    model, hist_dict = train_model(
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        args = args,
        DEVICE = DEVICE
    )     
    
    # dataframe of the history dict
    hist_df = pd.DataFrame(hist_dict)

    # save dataframe
    hist_df.to_csv(f'{output_csv_path}', index = False)

    # save model
    torch.save(model.state_dict(), f'{output_model_path}.pth')
