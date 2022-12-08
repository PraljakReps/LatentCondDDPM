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



class Att_enc_block(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
            ):
        super().__init__()
        self.Down_layer = Down(in_channels, out_channels)
        self.SelfAtt = SelfAttention(out_channels)


    def forward(self, x, t):
        h = self.Down_layer(x,t)
        return self.SelfAtt(h)


class Att_dec_block(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
        ):
        super().__init__()
        self.Up_layer = Up(in_channels, out_channels)
        self.SelfAtt = SelfAttention(out_channels)

    def forward(self, x, x_skip, t):

        h = self.Up_layer(x, x_skip, t)
        return self.SelfAtt(h)


class Attention_UNet(nn.Module):

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


    def forward(self, x, t = None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        
        return self.unet_forward(x, t)

