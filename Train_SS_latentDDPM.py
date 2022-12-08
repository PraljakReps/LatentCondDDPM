
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
from source.SemiSup_module import top_model, SemiSup_DDPM
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

        train_transforms = T.Compose(
                [
                    T.Resize([32,32]),
                    T.ToTensor(),
                    T.Normalize((0.5), (0.5)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5)
                ]
        )

        transforms = T.Compose(
                [
                    T.Resize([32,32]),
                    T.ToTensor(),
                    T.Normalize((0.5),(0.5))
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
    parser.add_argument('-dl', '--disc_num_layers', default = 2, help = 'flag: classifier depth', type = int)
    parser.add_argument('-dw', '--disc_width', default = 1000, help = 'flag: classifier width', type = int)



    # learning prefactors
    parser.add_argument('-al', '--alpha', default = 0.99, help = 'flag: mutual information (x,z) prefactor weight', type = float)
    parser.add_argument('-be', '--beta', default = 10, help = 'flag: MMD prefactor weight', type = float)
    parser.add_argument('-ga', '--gamma', default = 10, help = 'flag: discriminative prefactor weight', type = float)
    parser.add_argument('-x', '--xi', default = 1, help = 'flag: NLL prefactor weight', type = float)
       

    # filepaths
    parser.add_argument('-fop', '--fig_output_path', default = './output/SSlatentDDPM/mnist_samples_', help = 'flag: filenames for the mnist files', type = str)
    parser.add_argument('-omp', '--output_model_path', default = './output/SSlatentDDPM/model_pretrained_weights.pth', help = 'flag: model weights', type = str)
    parser.add_argument('-ocp', '--output_csv_path', default = './output/SSlatentDDPM/model_train_history.csv', help = 'flag: model training/testing loss history', type = str)
    parser.add_argument('-tfo', '--tsne_path', default = './output/SSlatentDDPM/tsne_plot', help = 'flag: ', type = str)

    args = parser.parse_args()

    return args



@torch.no_grad()
def end_epoch_validation(
        DEVICE: str,
        model: any,
        diffusion: any,
        test_dataloader: any
    ):
    
    model.eval()
    onehot_encoding = torch.eye(10)

    epoch_loss, epoch_mse_loss, epoch_kld_loss, epoch_mmd_loss, epoch_disc_loss = [],  [], [], [], []
    for iteration, valid_batch in enumerate(test_dataloader):
        
        # validation batch using the testing set
        valid_X0 = Variable(valid_batch[0].to(DEVICE))
        valid_y = Variable(onehot_encoding[
            valid_batch[1]].to(DEVICE)
        )
        
        # timesteps for each batch
        t = diffusion.sample_timesteps(valid_X0.shape[0]).to(DEVICE)    
            
        # corrupt training batch
        x_t, noise = diffusion.noise_images(valid_X0, t)

        # predict noise 
        predicted_noise, z, z_mean, z_var, y_pred = model(x_t, t, valid_X0)
        
        # sample from some noise:

        # normal dist:
        if DEVICE == 'cuda':
            z_true_samples = torch.randn_like(z).to(DEVICE)
        
        else:
            z_true_samples = torch.randn_like(z)
        
    
        # loss
        loss, loss_ddpm, loss_kld, loss_mmd, loss_disc = model.compute_loss(
                                        true_noise = noise,
                                        pred_noise = predicted_noise,
                                        z_true = z_true_samples,
                                        z_infer = z,
                                        z_mu = z_mean,
                                        z_var = z_var,
                                        y_true = valid_y,
                                        y_pred = y_pred
        )

        epoch_loss.append(loss.item())
        epoch_mse_loss.append(loss_ddpm.item())
        epoch_kld_loss.append(loss_kld.item())
        epoch_mmd_loss.append(loss_mmd.item())
        epoch_disc_loss.append(loss_disc.item())

    model.train()
    
    print(f'Validation | loss: {np.mean(epoch_loss)}'
        + f'| Mean squared error: {np.mean(epoch_mse_loss)}'
        + f'| KLD loss: {np.mean(epoch_kld_loss)}'
        + f'| MMD loss: {np.mean(epoch_mmd_loss)}'
        + f'| Disc loss: {np.mean(epoch_disc_loss)}'
    )


    return np.mean(epoch_loss), np.mean(epoch_mse_loss), np.mean(epoch_kld_loss), np.mean(epoch_mmd_loss), np.mean(epoch_disc_loss)
       

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
    disc_num_layers = args.disc_num_layers
    disc_width = args.disc_width

    # loss prefactors
    xi = args.xi
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

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

    tm_disc = top_model(
            num_layers = disc_num_layers,
            hidden_width = disc_width,
            z_dim = z_dim,
            num_classes = 10
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

    model = SemiSup_DDPM(
            Unet = Unet,
            latent_AE = rep_AE,
            top_model = tm_disc,
            diffusion = diffusion,
            xi = xi,
            alpha = alpha,
            beta = beta,
            gamma = gamma
    ).to(DEVICE)


    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)
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
        'test_mmd_loss': list(),
        'train_disc_loss': list(),
        'test_disc_loss': list()
    }
    
    onehot_encoding = torch.eye(10)

    # train model
    for epoch in range(args.epochs):
        
        epoch_loss, epoch_mse_loss, epoch_kld_loss, epoch_mmd_loss, epoch_disc_loss = [], [], [], [], []
        for iteration, train_batch in tqdm(enumerate(train_dataloader)):

            # training batch
            train_X0 = Variable(train_batch[0].to(DEVICE))
            train_y = Variable(
                onehot_encoding[train_batch[1]].to(DEVICE)
            )

            # timesteps for each batch
            t = diffusion.sample_timesteps(train_X0.shape[0]).to(DEVICE)

            # corrupt training batch
            x_t, noise = diffusion.noise_images(train_X0, t)

            # predicted noise
            predicted_noise, z, z_mean, z_var, y_pred = model(x = x_t, t = t, x0 = train_X0)
            
            # sample from some noise
            if DEVICE == 'cuda':
                z_true_samples = Variable(torch.randn_like(z)).to(DEVICE)
            else:
                z_true_samples = torch.randn_like(z)

            # loss
            loss, loss_ddpm, loss_kld, loss_mmd, loss_disc = model.compute_loss(
                                                    true_noise = noise,
                                                    pred_noise = predicted_noise,
                                                    z_true = z_true_samples,
                                                    z_infer = z,
                                                    z_mu = z_mean,
                                                    z_var = z_var,
                                                    y_true = train_y,
                                                    y_pred = y_pred
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_mse_loss.append(loss_ddpm.item())
            epoch_kld_loss.append(loss_kld.item())
            epoch_mmd_loss.append(loss_mmd.item())
            epoch_disc_loss.append(loss_disc.item())
            

        train_loss = np.mean(epoch_loss)
        train_mse_loss = np.mean(epoch_mse_loss)
        train_kld_loss = np.mean(epoch_kld_loss)
        train_mmd_loss = np.mean(epoch_mmd_loss)
        train_disc_loss = np.mean(epoch_disc_loss)

        print(f'Training | loss: {train_loss} '
            + f'| Mean squared error: {train_mse_loss} ' 
            + f'| KLD loss: {train_kld_loss} ' 
            + f'| MMD loss: {train_mmd_loss}'
            + f'| disc loss: {train_disc_loss}'
        )
        
        history_dict['epoch'].append(epoch)
        history_dict['train_loss'].append(train_loss)
        history_dict['train_mse_loss'].append(train_mse_loss)
        history_dict['train_kld_loss'].append(train_kld_loss)
        history_dict['train_mmd_loss'].append(train_mmd_loss)
        history_dict['train_disc_loss'].append(train_disc_loss)
        
        test_loss, test_mse_loss, test_kld_loss, test_mmd_loss, test_disc_loss = end_epoch_validation(
            DEVICE = DEVICE,
            model = model,
            diffusion = diffusion,
            test_dataloader = test_dataloader
        )
    
        history_dict['test_loss'].append(test_loss)
        history_dict['test_mse_loss'].append(test_mse_loss)
        history_dict['test_kld_loss'].append(test_kld_loss)
        history_dict['test_mmd_loss'].append(test_mmd_loss)
        history_dict['test_disc_loss'].append(test_disc_loss)

        # generate samples to quantify quality:
        sample_images = hf.generate_sample(
                args = args,
                task = args.tasks,
                diffusion = diffusion,
                model = model,
                n = 20,
                DEVICE = DEVICE
        )
        hf.plot_random_samples(
                args = args,
                sample_images = sample_images)
        plt.savefig(f'{args.fig_output_path}[epoch={epoch}].png', dpi = 300)
 

    return model, history_dict


if __name__ == '__main__':

    # get arguments
    args = get_args()
    
    SEED = args.SEED # random seed
    batch_size = args.batch_size # batch size
    output_model_path = args.output_model_path # model output path
    output_csv_path = args.output_csv_path # train/test loss history path
    task = args.tasks # pick data

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
                                            task=task,
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
