
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

from source.vanilla_module import Attention_UNet
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
    parser.add_argument('-t', '--task', default = 'mnist', help = 'flag: dataset task', type = str)
    # model hyperparameter
    parser.add_argument('-cin', '--c_in', default = 1, help = 'flag: input channel to the Unet model', type = int)
    parser.add_argument('-cout', '--c_out', default = 1, help = 'flag: output channel to the Unet model', type = int)
    parser.add_argument('-fnc', '--first_num_channel', default = 64, help = 'flag: number of conv channels for the first layer', type = int)
    parser.add_argument('-td', '--time_dim', default = 256, help = 'flag: embedding dimension for the time positions', type = int)
    parser.add_argument('-nl', '--num_layers', default = 3, help = 'flag: number of layers', type = int)
    parser.add_argument('-bn', '--bn_layers', default = 2, help = 'flag: number of Unets bottleneck layers', type = int)
    parser.add_argument('-rl', '--rep_learning', default = False, help = 'flag: choose to use latent conditional info or not.', type = bool)


    # filepaths
    parser.add_argument('-fop', '--fig_output_path', default = './output/vanillaDDPM/mnist_samples_', help = 'flag: filenames for the mnist files', type = str)
    parser.add_argument('-omp', '--output_model_path', default = './output/vanillaDDPM/model_pretrained_weights.pth', help = 'flag: model weights', type = str)
    parser.add_argument('-ocp', '--output_csv_path', default = './output/vanillaDDPM/model_train_history.csv', help = 'flag: model training/testing loss history', type = str)

    args = parser.parse_args()

    return args

@torch.no_grad()
def sample(
        args,
        diffusion: any,
        model: any,
        n: int = 20
    ):
    
    model.eval()
    
        
    x_sample = torch.randn((n, args.c_in, diffusion.img_size, diffusion.img_size)).to(diffusion.device)
        
    for ii in tqdm(reversed(range(1, diffusion.noise_steps)), position = 0):
            
        t = (torch.ones(n) * ii).long().to(diffusion.device)
            
        # predict noise
        predicted_noise = model(x_sample, t)
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
    
    
    if args.task == 'mnist':
        return x_sample

    else:
        x_sample = (x_sample*0.5+0.5).permute(0,2,3,1)
        return x_sample



def plot_samples(
        args: any,
        fig_output_path: str,
        X_sample: torch.FloatTensor,
        epoch: int
    ):

    fig, axes = plt.subplots(5,4, dpi = 300, sharex = True, sharey = True)

    fig.subplots_adjust(hspace = 0, wspace = 0)

    idx = 0
    for ii in range(5):

        for jj in range(4):

            if args.task == 'mnist':
                axes[ii,jj].imshow(
                        X_sample[idx,0].cpu()
                )
            else:
                axes[ii,jj].imshow(
                        X_sample[idx,:,:,:].cpu()
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
    ) -> list:

    model.eval()
    mse = nn.MSELoss(reduction = 'none')

    mse_loss = []
    for iteration, test_batch, in enumerate(test_dataloader):

        # validation batch using the testing set
        test_batch = Variable(test_batch[0].to(DEVICE))
        
        # timesteps for each batch
        t = diffusion.sample_timesteps(test_batch.shape[0]).to(DEVICE)

        # corrupt testing batch
        x_t, noise = diffusion.noise_images(test_batch, t)

        # predict noise
        predicted_noise = model(x_t, t)

        # MSE loss, where we sum over pixels and average over batch
        loss = torch.mean(
                torch.sum( mse(noise, predicted_noise), dim = (-2, -1)
                )
        )

        mse_loss.append(loss.item())
    model.train()

    print('MSE loss for the validation set:', np.mean(mse_loss))

    return np.mean(mse_loss)
        

def train_model(
    train_dataloader: any,
    test_dataloader: any,
    args: any,
    DEVICE: str = 'cuda'
    ) -> (any, dict):
    
    # model hyperparameters
    c_in = args.c_in
    c_out = args.c_out
    first_num_channel = args.first_num_channel
    time_dim = args.time_dim
    num_layers = args.num_layers
    bn_layers = args.bn_layers

    # configure model
    model = Attention_UNet(
                    c_in = c_in,
                    c_out = c_out,
                    first_num_channel = first_num_channel,
                    time_dim = time_dim,
                    num_layers = num_layers,
                    bn_layers = bn_layers
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)
    mse = nn.MSELoss(reduction = 'none')
    diffusion = Diffusion(
        img_size =  next(iter(train_dataloader))[0].shape[-1],
        device = DEVICE,
        schedule = 'linear'
    )

    model.train()
    
    history_dict = {
        'epoch': list(),
        'train_MSE': list(),
        'test_MSE': list()
    }

    # train model
    for epoch in range(args.epochs):
        
        epoch_loss = []
        for iteration, train_batch in tqdm(enumerate(train_dataloader)):

            # training batch
            train_batch = Variable(train_batch[0].to(DEVICE))

            # timesteps for each batch
            t = diffusion.sample_timesteps(train_batch.shape[0]).to(DEVICE)

            # corrupt training batch
            x_t, noise = diffusion.noise_images(train_batch, t)

            # predicted noise
            predicted_noise = model(x = x_t, t = t)

            # MSE loss, where we sum over pixels and average over batch
            loss = torch.mean(
                    torch.sum(
                        mse(noise, predicted_noise), dim = (-2,-1)
                    )
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
       

        train_MSE_loss = np.mean(epoch_loss)
        print('Training MSE:', train_MSE_loss)

        history_dict['epoch'].append(epoch)
        history_dict['train_MSE'].append(train_MSE_loss)
        

        test_MSE_loss = end_epoch_validation(
            DEVICE = DEVICE,
            model = model,
            diffusion = diffusion,
            test_dataloader = test_dataloader
        )

        history_dict['test_MSE'].append(test_MSE_loss)
    
        # generate samples at the end of epoch
        X_sample = sample(
                args = args,
                diffusion = diffusion,
                model = model,
                n = 20
        )
       
        # save generate samples
        plot_samples(
            args = args,
            fig_output_path = args.fig_output_path,
            X_sample = X_sample,
            epoch = epoch
        )

    return model, history_dict


if __name__ == '__main__':

    # get arguments
    args = get_args()
    
    SEED = args.SEED # random seed
    batch_size = args.batch_size # batch size
    output_model_path = args.output_model_path # model output path
    output_csv_path = args.output_csv_path # train/test loss history path
    task = args.task # problem task: mnist vs cifar10

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
