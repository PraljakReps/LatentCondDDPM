
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE
from tqdm import tqdm 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sample(args, diffusion, model, n, device):
    mse = nn.MSELoss(reduction = 'none')
    
    model.eval()

    with torch.no_grad():
        
        x = torch.randn((n, 1, diffusion.img_size, diffusion.img_size)).to(device)           
        #x_0 = torch.zeros((n, 1, diffusion.img_size, diffusion.img_size)).to(device)
        z_0 = torch.randn((n, args.z_dim)).to(device)
        
        for ii in tqdm(reversed(range(1, diffusion.noise_steps)), position = 0):
            
            t = (torch.ones(n) * ii).long().to(device)
            
            # corrupt training batch
            x_z = model.sample_randomly(z_0, t)
            
            # predict noise
            predicted_noise = model.Unet(x, x_z, t)
            
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if ii > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
                
            alpha_prefactor = (1-alpha)/ (torch.sqrt(1-alpha_hat))

            x = 1 / torch.sqrt(alpha) * (x - alpha_prefactor*predicted_noise) + torch.sqrt(beta) * noise
            
    mse_loss = torch.mean(
        torch.sum(
        mse(noise, predicted_noise), dim = (-2, -1)
        )
    )
    print(f'MSE loss: {mse_loss}')
            
    model.train()
    x = (x.clamp(-1, 1) + 1)/ 2
    return x
                


def cifar_sample(args, diffusion, model, n, device):
    mse = nn.MSELoss(reduction = 'none')
    
    model.eval()

    with torch.no_grad():
        
        x = torch.randn((n, 3, diffusion.img_size, diffusion.img_size)).to(device)           
        #x_0 = torch.zeros((n, 1, diffusion.img_size, diffusion.img_size)).to(device)
        z_0 = torch.randn((n, args.z_dim)).to(device)
        
        for ii in tqdm(reversed(range(1, diffusion.noise_steps)), position = 0):
            
            t = (torch.ones(n) * ii).long().to(device)
            
            # corrupt training batch
            x_z = model.sample_randomly(z_0, t)
            
            # predict noise
            predicted_noise = model.Unet(x, x_z, t)
            
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if ii > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
                
            alpha_prefactor = (1-alpha)/ (torch.sqrt(1-alpha_hat))

            x = 1 / torch.sqrt(alpha) * (x - alpha_prefactor*predicted_noise) + torch.sqrt(beta) * noise
            
    mse_loss = torch.mean(
        torch.sum(
        mse(noise, predicted_noise), dim = (-2, -1)
        )
    )
    print(f'MSE loss: {mse_loss}')
            
    model.train()
    x = (x.clamp(-1, 1) + 1)/ 2

    # unnormalize
    x = (x*0.5 + 0.5).permute(0,2,3,1)

    return x
                


def generate_sample(
        args,
        task,
        diffusion,
        model,
        n,
        DEVICE
    ):

    if task == 'mnist':
        return sample(args, diffusion, model, 20, DEVICE)

    else:
        return cifar_sample(args, diffusion, model, 20, DEVICE)


def plot_random_samples(args, sample_images):
    
    fig, axes = plt.subplots(5, 4, dpi = 300, sharex = True, sharey = True)

    fig.subplots_adjust(hspace=0, wspace = 0)


    idx = 0
    for ii in range(5):

        for jj in range(4):


            if args.tasks == 'mnist':

                axes[ii, jj].imshow(
                    sample_images[idx,0].cpu()
                )

            else:
                
                axes[ii, jj].imshow(
                        sample_images[idx,:,:,:].cpu()
                )

            axes[ii,jj].axis('off')

            axes[ii,jj].set_xticklabels([])
            axes[ii,jj].set_yticklabels([])
            idx+=1 


    plt.tight_layout()


def infer_latents(
    train_dataloader,
    model,
    img_size,
    batch_size,
    device
    ):
    
    num_train_iters = len(train_dataloader) # number of minibatch iterations
    num_train_samples = num_train_iters * batch_size
    
    model.eval()

    # AE reconstructions (conditionals)
    AE_x_pred = torch.zeros((num_train_samples, 1, img_size, img_size))
    # predicted latent codes:
    z_pred = torch.zeros((num_train_samples, model.latent_AE.z_dim))
    # ground truth class labels:
    y_train = torch.zeros((num_train_samples, 1))

    with torch.no_grad():

        left_idx = 0
        right_idx = batch_size
        for train_batches in tqdm(train_dataloader):

            # training batch
            train_x = train_batches[0]
            train_y = train_batches[1]
            n = train_x.shape[0]
            t = (torch.ones(n) * 0).long().to(device)

            x_temp, z_temp, _, _ = model.latent_AE(train_x.to(device), t)

            AE_x_pred[left_idx:right_idx, :, :, :] = x_temp.cpu()
            z_pred[left_idx:right_idx, :] = z_temp.cpu()
            y_train[left_idx:right_idx, :] = train_y.cpu().unsqueeze(1)

            left_idx+=batch_size
            right_idx+=batch_size

        model.train()
        
    return AE_x_pred, z_pred, y_train


def plot_tsne(z_pred, y_train):
    
    from sklearn.manifold import TSNE

    tsne_z_pred = TSNE(n_components=2, perplexity=50).fit_transform(z_pred[0:1000])
                                                                           
    fig, axes = plt.subplots(1, 2, dpi = 300)
    
    axes[0].scatter(
        tsne_z_pred[:,0],
        tsne_z_pred[:,1],
        c = y_train[0:1000],
        edgecolor = 'k',
        cmap = 'jet'
    )
                                                                           
    
    axes[1].scatter(
    z_pred[0:5000,0].detach().cpu().numpy(),
    z_pred[0:5000,1].detach().cpu().numpy(),
    c = y_train[:5000], 
    cmap = 'jet'
    )
 
    axes[0].set_xlabel('tsne 0')
    axes[0].set_ylabel('tsne 1')

    axes[1].set_xlabel('$z_0$')
    axes[1].set_ylabel('$z_1$')

    plt.tight_layout()




def plot_AE_decodings(AE_x_pred):
    
    fig, axes = plt.subplots(5, 4, dpi = 300, sharex = True, sharey = True)

    fig.subplots_adjust(hspace=0, wspace = 0)


    idx = 0
    for ii in range(5):

        for jj in range(4):

            axes[ii, jj].imshow(
                 AE_x_pred[idx,0].cpu()
            )
            axes[ii,jj].axis('off')

            axes[ii,jj].set_xticklabels([])
            axes[ii,jj].set_yticklabels([])
            idx+=1 


    plt.tight_layout()
    plt.show()

