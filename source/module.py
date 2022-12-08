import torch
import torch.nn as nn
import torch.nn.functional as F



def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class DoubleConv(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            mid_channels = None,
            residual = None
    ):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
                nn.GroupNorm(1, mid_channels),
                nn.GELU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
                nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):

        if self.residual:
            return F.gelu(x + self.double_conv(x))

        else:
            return self.double_conv(x)


class Down(nn.Module):

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
                    out_channels = out_channels,
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


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
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

    def forward(self, x, skip_x, t):

        x = self.up(x)

        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):

    def __init__(self, channels):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first = True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
                nn.LayerNorm([channels]),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels)
                )

    def forward(self, x):

        size = x.shape[-1]
        x = x.view(-1, self.channels, size*size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.channels, size, size)



class UNet(nn.Module):

    def __init__(self, c_in = 3, c_out = 3, time_dim = 256):
        super().__init__()

        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)


        # Unet's bottleneck
        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)


        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size = 1)

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

        x1 = self.inc(x)
        
        # downscale
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        

        # bottleneck
        bot_x1 = self.bot1(x4)
        bot_x2 = self.bot2(bot_x1)
       
        # upscale 
        x_up = self.up1(bot_x2, x3, t)
        x_up = self.up2(x_up, x2, t)
        x_up = self.up3(x_up, x1, t)
        
        # output (reconstruction)
        output = self.outc(x_up)
        return output


    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)



class Attention_UNet(nn.Module):

    def __init__(self, c_in = 3, c_out = 3, time_dim = 256):
        super().__init__()

        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64*2)
        self.down1 = Down(64*2, 128*2)
        self.sa1 = SelfAttention(128*2)
        self.down2 = Down(128*2, 256*2)
        self.sa2 = SelfAttention(256*2)
        self.down3 = Down(256*2, 256*2)
        self.sa3 = SelfAttention(256*2)


        # Unet's bottleneck
        self.bot1 = DoubleConv(256*2, 256*2)
        self.bot2 = DoubleConv(256*2, 256*2)


        self.up1 = Up(512*2, 128*2)
        self.sa4 = SelfAttention(128*2)
        self.up2 = Up(256*2, 64*2)
        self.sa5 = SelfAttention(64*2)
        self.up3 = Up(128*2, 64*2)
        self.sa6 = SelfAttention(64*2)
        self.outc = nn.Conv2d(64*2, c_out, kernel_size = 1)

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


    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)



class Attention_CIFAR_UNet(nn.Module):

    def __init__(self, num_layers, num_bn_layers, c_in = 3, c_out = 3, time_dim = 256):
        super().__init__()

        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)


        self.down_sampler = nn.ModuleList()

        # encoder path
        in_channels = 64
        for ii in range(num_layers):

            out_channels = in_channels * 2

            self.down_sampler.append(
                    Down(in_channels, out_channels)
            )
            self.down_sampler.append(
                    SelfAttention(out_channels)
            )
            in_channels = out_channels
        
        # Unet's bottleneck
        self.bottleneck = nn.ModuleList()
        for ii in range(num_bn_layers):
            self.bottleneck.append(DoubleConv(out_channels, out_channels))

        
        # decoder path:
        self.up_sampler = nn.ModuleList()
        in_channels = 512
        for ii in range(num_layers):

            out_channels = in_channels // 4
           
            if out_channels < 64:
                out_channels = in_channels // 2
            else:
                pass 

            # make lower bound for lower channels
            self.up_sampler.append(
                    Up(in_channels, out_channels)
            )
            self.up_sampler.append(
                    SelfAttention(out_channels)
            )
        
            in_channels = out_channels * 2 

        self.outc = nn.Conv2d(64, c_out, kernel_size = 1)

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

        x_down = self.inc(x)
        
        encoder_skips = []

        # downscale
        for layer in self.down_sampler:
            
            try:
                # check whether the layer contains MHA
                if isinstance(layer.mha, nn.MultiHeadAttention):
                    x_down = layer(x_down)
                else:
                    x_down = layer(x_down, t)
                    encoder_skips.append(x_down)

            except:
                x_down = layer(x_down, t)
                encoder_skips.append(x_down)

        # bottleneck
        bot_x = x_down.clone()
        for bn_layer in self.bottleneck:
            bot_x = bn_layer(bot_x)


        # upscale
        x_up = bot_x.clone()
        skip_use = 1
        for layer in self.up_sampler:

            try:
                # check whether the layer contains MHA
                if isinstance(layer.mha, nn.MultiHeadAttention):
                    x_up = layer(x_up)
                else:
                    x_up = layer(x_up, encoder_skips[-skip_use], t)
                    skip_use += 1
            except:
                x_up = layer(x_up, encoder_skips[-skip_use], t)
                skip_use += 1
 

        # output (reconstruction)
        output = self.outc(x_up)
        return output


    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)



