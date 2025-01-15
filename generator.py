from tkinter import S
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F
import re
from einops import rearrange
import numbers


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # print('x.shape', x.shape)
        # print('mu.shape', mu.shape)

        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

# The code was inspired from https://github.com/LMescheder/GAN_stability.

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, label_nc=3):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt:
            self.conv_0 = spectral_norm(self.conv_0)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, label_nc)


    def forward(self, x, seg):

        x_s = self.shortcut(x, seg)
        dx = self.actvn(self.norm_0(x, seg))
        dx = self.conv_0(dx)
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    



##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):

    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)

        

    def forward(self,x):
        B,C,H,W= x.shape
        emb = x.mean(dim=(-2,-1))
        emb = self.linear_layer(emb)
        prompt_weights = F.softmax(emb-emb.max(),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        # noise = torch.randn(prompt.size()).to(prompt.device) * 0 + 0
        # prompt = prompt + noise
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="nearest")
        prompt = self.conv3x3(prompt)
        return prompt



##########################################################################
##---------- Prompt Inj Module -----------------------
class PromptInjBlock(nn.Module):

    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptInjBlock,self).__init__()
        self.prompt =  PromptGenBlock(prompt_dim,prompt_len,prompt_size,lin_dim)
        self.transformer = TransformerBlock(dim=prompt_dim*2, num_heads=4, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias')
        self.conv3x3 = nn.Conv2d(prompt_dim*2,prompt_dim,kernel_size=1,bias=False)


    def forward(self,x, seg):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        prompt = self.prompt(seg)
        x = torch.cat([x, prompt], 1)
        x = self.transformer(x)
        x = self.conv3x3(x)
        return x



 ##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.scale = (dim // num_heads) ** -0.5
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        ## prevent overflow, especially for fp 16 training
        attn = (attn-attn.max()).softmax(dim=-1)
        ## prevent underflow, especially for fp 16 training
        dtype_min = torch.tensor(torch.finfo(attn.dtype).min, device=attn.device,dtype=attn.dtype)
        attn = torch.max(attn, dtype_min)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
   


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x




class Up_ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2,False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        '''self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            activation)'''

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size)),
            activation,
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size)),
            activation
        )
        

    def forward(self, x):
        y = self.conv_block(x)
        return y
    
    
class HINet_1(nn.Module):

    def __init__(self, in_chn=3, wf=16, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(HINet_1, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.ad1_list = nn.ModuleList()
        self.ad2_list = nn.ModuleList()
        self.tanh = nn.Tanh()
        prev_channels = self.get_input_chn(wf)
        norm_G = "spectralspadesyncbatch3x3"
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            self.ad2_list.append(PromptInjBlock((2**i) * wf, 5, (2**i) * wf, (2**i) * wf))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.ad1_list = self.ad1_list[0:-1]
        self.ad2_list = self.ad2_list[0:-1]

        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))            
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, 1, bias=True)

        

    def forward(self, x, latent_list):
        image = x
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)                 
                encs.append(x1_up)
            else:
                x1 = down(x1)

                

        for i, up in enumerate(self.up_path_1):
            temps2 = self.ad1_list[-1-i](encs[-i-1], latent_list[-1-i])
            temps3 = self.ad2_list[-1-i](encs[-i-1], latent_list[-1-i])
            temps4 = temps2 + temps3
            x1 = up(x1, temps4)
            decs.append(x1)
        out = self.last(x1)
        out = out + image[:,1,:,:].unsqueeze(dim=1)
        # make fp 16 training happy 
        if out.dtype == torch.float16:
            max_dtype = torch.finfo(out.dtype).max
            min_dtype = torch.finfo(x.dtype).min
            clamp_value = torch.where(torch.isinf(x).any(), max_dtype - 1000, max_dtype)
            clamp_value_1 = torch.where(torch.isnan(x).any(), min_dtype, min_dtype + 1000)
            out = torch.clamp(out, min=-clamp_value, max=clamp_value)
            out = torch.clamp(out, min=-clamp_value_1, max=clamp_value_1)
        out = self.tanh(out)
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class HINet_2(nn.Module):

    def __init__(self, in_chn=3, wf=16, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(HINet_2, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.ad1_list = nn.ModuleList()
        self.tanh = nn.Tanh()
        prev_channels = self.get_input_chn(wf)
        norm_G = "spectralspadesyncbatch3x3"
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.ad1_list = self.ad1_list[0:-1]

        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))            
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, 1, bias=True)

        

    def forward(self, x, latent_list):
        image = x
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)               
                encs.append(x1_up)
            else:
                x1 = down(x1) 
                

        for i, up in enumerate(self.up_path_1):
            temps2 = self.ad1_list[-1-i](encs[-i-1], latent_list[-1-i])
            x1 = up(x1, temps2)
            decs.append(x1)
        out = self.last(x1)
        out = out + image[:,1,:,:].unsqueeze(dim=1)
        # make fp 16 training happy 
        if out.dtype == torch.float16:
            max_dtype = torch.finfo(out.dtype).max
            min_dtype = torch.finfo(x.dtype).min
            clamp_value = torch.where(torch.isinf(x).any(), max_dtype - 1000, max_dtype)
            clamp_value_1 = torch.where(torch.isnan(x).any(), min_dtype, min_dtype + 1000)
            out = torch.clamp(out, min=-clamp_value, max=clamp_value)
            out = torch.clamp(out, min=-clamp_value_1, max=clamp_value_1)
        out = self.tanh(out)
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        # TODO
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc