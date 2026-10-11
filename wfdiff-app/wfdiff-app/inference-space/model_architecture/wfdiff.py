"""
WF-Diff architecture and inference helpers.
Extracted from WF-Diff_UIEB_hybrid.ipynb (Zhao et al., CVPR 2024).
"""
import math
import numbers
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

USE_GRAD_CHECKPOINTING = False

# ============================================================================
# WF-Diff  (Zhao et al., CVPR 2024)  --  github.com/ChenzhaoNju/WF-Diff
# Cell 1/2 : building blocks, copied from the repository's basicsr/archs/ sources.
#   Only mechanical changes (see the notes in the markdown cell above):
#     * BasicSR registry decorators dropped
#     * two different classes that share a name across repo files are renamed
#         trans_block_eca.FeedForward      -> FeedForward_UNet
#         dft_arch.FeedForward             -> FeedForward_DFT
#         unet_block/cross_attention_module.CFC -> CFC_UNet   (cfc_arch.CFC keeps the name CFC)
#     * identical duplicates (DWT/IWT, Depth_conv, LayerNorm, ...) are defined once
#   Attribute names are untouched, so state_dict keys are identical to the repository's.
# ============================================================================
import numbers
from functools import partial
from inspect import isfunction
from einops import rearrange

# ----------------------------------------------------------------------------
# Haar wavelet  (Padiff_arch/wavelet.py)
# ----------------------------------------------------------------------------

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# ----------------------------------------------------------------------------
# Shared blocks  (Depth_conv / LayerNorm are byte-identical in the repo's files)
# ----------------------------------------------------------------------------

class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ----------------------------------------------------------------------------
# Denoiser blocks  (Padiff_arch/unet_block/trans_block_eca.py, cross_attention_module.py, unetx2_arch.py)
# ----------------------------------------------------------------------------

class FeedForward_UNet(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_UNet, self).__init__()

        self.x1_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x1_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x2_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x2_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x3_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x3_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x4_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x4_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x5_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x5_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)

        self.outCon = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        x1 = self.x1_Conv5(self.x1_Conv1(x))
        x2 = self.x2_Conv5(self.x2_Conv1(x))
        x3 = self.x3_Conv5(self.x3_Conv1(x))

        x1 = x2 + self.x4_Conv5(self.x4_Conv1(F.gelu(x1) * x2))
        x3 = x2 + self.x5_Conv5(self.x5_Conv1(F.gelu(x3) * x2))

        return self.outCon(torch.concat([x1,x3],dim=1))


class PA_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias,t_dim = 3):
        super(PA_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.q_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        
        self.k_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        
        k = self.k_dwconv(self.k(x))
       
        v = self.v_dwconv(self.v(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
       
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
       
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = out + x

        # out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, with_PPU=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.atten = PA_MSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_UNet(dim, ffn_expansion_factor, bias)
        self.with_PPU = with_PPU

    def forward(self, x, time):
        x = x + self.atten(self.norm1(x)+time)
        x = x + self.ffn(self.norm2(x)+time)
        return x


class CFC_UNet(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.2):
        super(CFC_UNet, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        # self.valueh = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):

        # q -> transmission map
        mixed_query_layer = self.query(hidden_states)
        # K -> Xc A Xt 的 concat
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ----------------------------------------------------------------------------
# WFI2-net blocks  (Padiff_arch/dft_arch.py)
# ----------------------------------------------------------------------------

class FeedForward_DFT(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward_DFT, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class DTB(nn.Module):

    def __init__(self, dim, num_heads, ffn_factor, bias, LayerNorm_type):
        super(DTB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_DFT(dim, ffn_factor, bias)

    def forward(self, x):
       
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.valueh = Depth_conv(in_ch=dim, out_ch=dim)
        self.valuel = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        n, c, h, w = hidden_states.shape
        ctx1 = ctx[:n, ...]
        ctx2 =  ctx[n:n+n, ...]
        ctx3 =  ctx[n+n:, ...]
        ctx=ctx1+ctx2+ctx3
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layerh = self.valueh(ctx)
        mixed_value_layerl = self.valuel(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layerh = self.transpose_for_scores(mixed_value_layerh)
        value_layerl = self.transpose_for_scores(mixed_value_layerl)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layerh = torch.matmul(attention_probs, value_layerh)
        ctx_layerh = ctx_layerh.permute(0, 2, 1, 3).contiguous()

        ctx_layerl = torch.matmul(attention_probs, value_layerl)
        ctx_layerl = ctx_layerl.permute(0, 2, 1, 3).contiguous()

        return ctx_layerh,ctx_layerl


class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        #self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              #bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SRDB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SRDB, self).__init__()
        nChannels_ = nChannels
        modules1 = []
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                              bias=False)
        
        #self.conv11 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
        #                      bias=False)
        #self.conv22 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
        #                      bias=False)
        #self.conv33 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
        #                      bias=False)
        
        #self.conv4 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
         #                     bias=False)
        #self.conv5 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
        #                      bias=False)
        self.conv6 = nn.Conv2d(growthRate*3, nChannels, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.leaky1=nn.LeakyReLU(0.1,inplace=True)
        self.leaky2=nn.LeakyReLU(0.1,inplace=True)
        self.leaky3=nn.LeakyReLU(0.1,inplace=True)
        #self.bat1 = nn.BatchNorm2d(nChannels),
        #self.bat2 = nn.BatchNorm2d(nChannels),
        #self.bat3 = nn.BatchNorm2d(nChannels),
        #self.bat4 = nn.BatchNorm2d(nChannels),
        #self.bat5 = nn.BatchNorm2d(nChannels),
        #self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
                                              #embed_dim=embed_dims[0])


    def forward(self, x):
        #x_1=self.bat1(self.conv1(x))
        x_1= self.leaky1(self.conv1(x))
        x_2= self.leaky2(self.conv2(x))
        x_3= self.leaky3(self.conv3(x))
        x_0=torch.cat((x_1,x_2,x_3),dim=1)
        #print(x_0.shape)

        #x_11=x_1+x_3
        #x_22=x_1+x_3+x_2
        #x_33=x_2+x_3

        #x_111= self.conv11(x_11)
        #x_222= self.conv22(x_22)
        #x_333= self.conv33(x_33)

        #x111=x_111*x_222+x_111
        #x333=x_222*x_333+x_333

        #x_o1= self.conv4(x111)
        #x_02= self.conv5(x333)

        #x_0=x_o1+x_02+x_1+x_3

        #x_0=self.conv6(x_0)
        #x_0=x_111+x+x_222+x_333
        x_0=self.conv6(x_0)

        out = x_0 + x
        return out


class FRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(FRDB, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)
        self.SRDB=SRDB(nChannels)
        #self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
                                              #embed_dim=embed_dims[0])


    def forward(self, x):
        x=self.SRDB(x)
        _, _, H, W = x.shape

        # Keep FFT + complex reconstruction in FP32 for numerical stability.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_freq = torch.fft.rfft2(x.float(), norm='backward')
            mag = torch.abs(x_freq)
            pha = torch.angle(x_freq)

        # Learnable convolutional blocks remain eligible for FP16 AMP.
        mag = self.dense_layers1(mag)
        mag = self.conv_1(mag)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)

        # Keep trigonometric + complex construction + inverse FFT in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            mag = mag.float()
            pha = pha.float()
            real = mag * torch.cos(pha)
            imag = mag * torch.sin(pha)
            x_out = torch.complex(real, imag)
            out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
            out = out + x.float()

        return out


# ============================================================================
# WF-Diff  --  Cell 2/2 : WFI2-net (DFTHL1), CFC, diffusion UNet, Gaussian diffusion, and the top-level WfDiffx2
# ============================================================================

# ----------------------------------------------------------------------------
# WFI2-net  (Padiff_arch/dft_arch.py : DFTHL1)
# ----------------------------------------------------------------------------

class DFTHL1(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_factor = 4.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(DFTHL1, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 =  FRDB(nChannels=dim) 
        #self.encoder_level1 = MySequential(*[
            #DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             #LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level11 = MySequential(*[
            DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
         


        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_21 = Downsample(dim)
        
        self.encoder_level2 = FRDB(nChannels=dim * 2 ** 1)
        #self.encoder_level2 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level21 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_31 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = FRDB(nChannels=dim * 2 ** 2)
        #self.encoder_level3 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level31 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_41 = Downsample(int(dim * 2 ** 2))

        self.latent = FRDB(nChannels=dim * 2 ** 2)
        #self.latent = MySequential(*[
            #DTB(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent1 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[3], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.cross_attention0 = cross_attention(dim=int(dim * 2 ** 2), num_heads=8)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_31 = Upsample(int(dim * 2 ** 3))

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level31 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = FRDB(nChannels=dim * 2 ** 2)
        #self.decoder_level3 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level31 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_21 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level21 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = FRDB(nChannels=dim * 2 ** 1)
        #self.decoder_level2 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level21 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_11 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = FRDB(nChannels=dim * 2 ** 1)
        #self.decoder_level1 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level11 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output1 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.upH1 = Upsample(int(dim * 2 ** 2))
        self.upL1 = Upsample(int(dim * 2 ** 2))
        # self.upL2 = Upsample(int(dim * 2 ** 2))
        # self.upH2 = Upsample(int(dim * 2 ** 2))

        

    def forward(self, inp_img):

        dwt,idwt= DWT(),IWT()

        input_img = inp_img[:, :3, :, :]
        n, c, h, w = input_img.shape
    
        input_dwt = dwt(input_img)
        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]


        inp_enc_level1 = self.patch_embed(input_LL)
        #out_enc_level1,_ = self.encoder_level1(inp_enc_level1, t)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        #out_enc_level2,_ = self.encoder_level2(inp_enc_level2, t)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        #out_enc_level3,_ = self.encoder_level3(inp_enc_level3, t)
        
        
        
        # out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # inp_enc_level4 = self.down3_4(out_enc_level3)


        inp_enc_level11 = self.patch_embed1(input_high0)
        out_enc_level11 = self.encoder_level11(inp_enc_level11)
        inp_enc_level21 = self.down1_21(out_enc_level11)
        out_enc_level21 = self.encoder_level21(inp_enc_level21)
        inp_enc_level31 = self.down2_31(out_enc_level21)


        # out_enc_level31 = self.encoder_level31(inp_enc_level31)
        # inp_enc_level41 = self.down3_41(out_enc_level31) 
        # x_HH,x_LL = self.cross_attention0(inp_enc_level4, inp_enc_level41)
        # x_LL1 = self.upL1(x_LL)
        # x_HH1 = self.upH1(x_HH)

        x_HH,x_LL = self.cross_attention0(inp_enc_level3, inp_enc_level31)
        x_LL1 = self.upL1(x_LL)
        x_HH1 = self.upH1(x_HH)


        latent = self.latent(inp_enc_level3)
        #latent,_ = self.latent(inp_enc_level4, t)
        latent1 = self.latent1(inp_enc_level31)
        #print(latent.shape)
        #print(latent1.shape)
        #print( x_HH_LH.shape)
        # inp_dec_level3 = self.up4_3(latent+x_LL)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # #out_dec_level3,_ = self.decoder_level3(inp_dec_level3, t)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(latent+x_LL)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2= self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2+x_LL1)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        x_HH=torch.cat((x_HH,x_HH,x_HH),dim=0)
        # inp_dec_level31 = self.up4_31(latent1+x_HH)
        # inp_dec_level31 = torch.cat([inp_dec_level31, out_enc_level31], 1)
        # inp_dec_level31 = self.reduce_chan_level31(inp_dec_level31)
        # out_dec_level31= self.decoder_level31(inp_dec_level31)
        
        inp_dec_level21 = self.up3_21(latent1+x_HH)
        inp_dec_level21 = torch.cat([inp_dec_level21, out_enc_level21], 1)
        inp_dec_level21 = self.reduce_chan_level21(inp_dec_level21)
        out_dec_level21 = self.decoder_level21(inp_dec_level21)
        x_HH1=torch.cat((x_HH1,x_HH1,x_HH1),dim=0)
        inp_dec_level11 = self.up2_11(out_dec_level21+x_HH1)
        inp_dec_level11 = torch.cat([inp_dec_level11, out_enc_level11], 1)
        out_dec_level11 = self.decoder_level11(inp_dec_level11)

        #### For Dual-Pixel Defocus Deblurring Task ####
        
        ###########################
       
        out_dec_level11 = self.output(out_dec_level11)

        out_dec_level=idwt(torch.cat((out_dec_level1,out_dec_level11),dim=0))

        return out_dec_level,out_dec_level1,out_dec_level11


# ----------------------------------------------------------------------------
# CFC: cross-frequency fusion of the LL and high-frequency sub-bands  (Padiff_arch/cfc_arch.py)
# ----------------------------------------------------------------------------

class CFC(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(CFC, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.valueh = Depth_conv(in_ch=dim, out_ch=dim)
        self.valuel = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        n, c, h, w = hidden_states.shape
        ctx1 = ctx[:n, ...]
        ctx2 =  ctx[n:n+n, ...]
        ctx3 =  ctx[n+n:, ...]
        ctx=ctx1+ctx2+ctx3
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layerh = self.valueh(ctx)
        mixed_value_layerl = self.valuel(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layerh = self.transpose_for_scores(mixed_value_layerh)
        value_layerl = self.transpose_for_scores(mixed_value_layerl)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layerh = torch.matmul(attention_probs, value_layerh)
        ctx_layerh = ctx_layerh.permute(0, 2, 1, 3).contiguous()
        ctx_layerh = ctx_layerh.repeat(3, 1, 1, 1)

        ctx_layerl = torch.matmul(attention_probs, value_layerl)
        ctx_layerl = ctx_layerl.permute(0, 2, 1, 3).contiguous()

        return ctx_layerh,ctx_layerl


# ----------------------------------------------------------------------------
# Diffusion denoiser UNet  (Padiff_arch/unetx2_arch.py)
# ----------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class ResnetBloc_eca(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False,with_PPU = True):
        super().__init__()
        self.with_attn = with_attn
        if with_attn:
            #self.attn = ResidualBlock(dim, dim, dim_out, is_noise=True)
            self.attn = TransformerBlock(dim=int(dim), num_heads=dim, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias',with_PPU= with_PPU)
        
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        )
    def forward(self, x, time_emb, f_out):
        # print(x.shape)
        # print( f_out.shape)
        x=x+f_out
        
        time = self.mlp(time_emb).unsqueeze(2).unsqueeze(3)
        if self.with_attn :
            x = self.attn(x, time)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            in_channel=6,
            inner_channel=32,
            norm_groups=32,
    ):
        super().__init__()

        dim = inner_channel
        time_dim = inner_channel
        # x
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))
            # ,nn.PixelUnshuffle(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        
        self.conv1_t = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=False))
            # ,nn.PixelUnshuffle(2))
        self.conv2_t = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv3_t = nn.Sequential(
            nn.Conv2d(3, int(dim * 2 ** 1) // 8, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),nn.PixelUnshuffle(2))
        self.conv4_t = nn.Sequential(
            nn.Conv2d(3, int(dim * 2 ** 2) // 32, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),nn.PixelUnshuffle(2),nn.PixelUnshuffle(2))
        
        self.cam1 = CFC_UNet(dim,dim)
        self.cam2 = CFC_UNet(dim * 2 ** 1, dim * 2 ** 1)
        self.cam3 = CFC_UNet(dim * 2 ** 2, dim * 2 ** 2)
        self.cam4 = CFC_UNet(dim * 2 ** 3, dim * 2 ** 3)
        

        self.block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4 = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True,with_PPU=True)

        self.conv_up3 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 3), (dim * 2 ** 3) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_cat3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)
        self.conv_cat1 = nn.Conv2d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=False)

        self.decoder_block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)

    def forward(self, x, t, p_t):
        # 1
        x = self.conv1(x)
        # print(p_t.shape)
        # print(x.shape)
        f_out1 = self.conv1_t(p_t)
        # print(f_out1.shape)
        # print(x.shape)
        f_out1 = self.cam1(f_out1,x)
        x1 = self.block1(x, t, f_out1)

        # 2
        x2 = self.conv2(x1)
        f_out2 = self.conv2_t(p_t)
        f_out2 = self.cam2(f_out2,x2)
        x2 = self.block2(x2, t, f_out2)

        # 3
        x3 = self.conv3(x2)
        f_out3 = self.conv3_t(p_t)
        f_out3 = self.cam3(f_out3,x3)
        x3 = self.block3(x3, t, f_out3)

        # 4
        # x4 = self.conv4(x3)
        # f_out4 = self.conv4_t(p_t)
        # x4 = self.block4(x4, t, f_out4)

        # de_level3 = self.conv_up3(x4)
        # de_level3 = torch.cat([de_level3, x3], 1)
        # de_level3 = self.conv_cat3(de_level3)
        # de_level3 = self.decoder_block3(de_level3, t, f_out3)

        de_level2 = self.conv_up2(x3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2, t, f_out2)

        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, x1], 1)
        de_level1 = self.conv_cat1(de_level1)
        mid_feat = self.decoder_block1(de_level1, t, f_out1)

        return mid_feat


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        with_time_emb=True
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        dim = inner_channel

        self.encoder_water = Encoder(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)

        self.refine = ResnetBloc_eca(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=False)
        self.de_predict = nn.Sequential(nn.Conv2d(dim, out_channel, kernel_size=1, stride=1))


    def forward(self, x, time, p):

        t = self.time_mlp(time) if self.time_mlp is not None else None

        mid_feat = self.encoder_water(x, t, p)
        return self.de_predict(mid_feat)


# ----------------------------------------------------------------------------
# Gaussian diffusion (beta schedule, q-sampling, 10-step DDIM sampler)  (Padiff_arch/diffx2_arch.py)
# ----------------------------------------------------------------------------

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    # `a` is already a tensor in this implementation; avoid reconstructing it on every call.
    a_t = a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32, device=t.device)
    a_t = a_t.to(device=t.device)
    out = torch.gather(a_t, 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusionx2(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        with_time_emb=True,
        schedule_opt=None,
        sample_proc = 'ddim'
    ):
        super().__init__()
        self.denoise_fn = UNet( in_channel=in_channel,
                                out_channel=out_channel,
                                inner_channel=inner_channel,
                                norm_groups=norm_groups,
                                with_time_emb=with_time_emb,
                               )
        self.eta = 0
        self.sample_proc = sample_proc
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt)


    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        ddim_sigma = (self.eta * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (1 - alphas_cumprod / alphas_cumprod_prev)) ** 0.5)
        self.ddim_sigma = to_torch(ddim_sigma)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t,p_a,p_t, clip_denoised: bool, condition_x=None, style=None):
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t,p_a,p_t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t,p_a,p_t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_ddim2(self, x, t, t_next,inter, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        bt = extract(self.betas, t, x.shape)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)

        if condition_x is not None:
            et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t,inter)
        else:
            et = self.denoise_fn(x, t,inter)


        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        # x0_air_t = (x_air - et_air * (1 - at).sqrt()) / at.sqrt()
        if t_next == None:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - self.betas).cumprod(dim=0), t_next, x.shape)
        if self.eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            # xt_air_next = at_next.sqrt() * x0_air_t + (1 - at_next).sqrt() * et_air
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = self.eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)
            # xt_air_next = at_next.sqrt() * x0_air_t + c2 * et_air + c1 * torch.randn_like(x0_t)

        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0

        return xt_next

    @torch.no_grad()
    def p_sample_loop(self,lq,inter, cand=None):
        device = self.betas.device
        sample_inter = 10
        g_gpu = torch.Generator(device=device).manual_seed(44444)
        
        
        x = lq
        condition_x = lq
        shape = x.shape
        b = shape[0]
        img = torch.randn(shape, device=device, generator=g_gpu)
        ret_img = x

        
        time_steps = np.array([1898, 1491, 1136, 680, 340])
            # time_steps = np.asarray(list(range(0, 1000, int(1000/4))) + list(range(1000, 2000, int(1000/6))))
            # time_steps = np.flip(time_steps[:-1])
        for j, i in enumerate(time_steps):
            # print('i = ', i)
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if j == len(time_steps) - 1:
                t_next = None
            else:
                t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
            img = self.p_sample_ddim2(img, t, t_next, inter,condition_x=condition_x)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        
        return ret_img[-1], None

    @torch.no_grad()
    def test(self,lq,inter, continous=False, cand=None):
        return self.p_sample_loop(lq,inter, cand=cand )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), int(t), device=device, dtype=torch.long)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
    
    def p_losses(self,lq,gt,inter, noise=None):
        x_start = gt

        #condition_x = torch.concat([p_a,lq+nrn_out],dim=1)
        condition_x =lq

        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        x_recon = self.denoise_fn(
            torch.cat([condition_x, x_noisy], dim=1), t,inter)

        return noise, x_recon

    def forward(self,lq, gt,inter, *args, **kwargs):
        if gt != None :
            return self.p_losses(lq, gt,inter)
        else :
            return self.test(lq,inter)


# ----------------------------------------------------------------------------
# WF-Diff top level  (archs/wfdiffx2_arch.py : WfDiffx2)
#   forward(condition, gt)  -> training  (gt given):  (x_, noise_LL, pred_noise_LL, noise_high, pred_noise_high, AA, HH)
#   forward(condition, None)-> inference (gt=None):   (x_, residual_LL, None,       residual_high, None,       AA, HH)
# ----------------------------------------------------------------------------

class WfDiffx2(nn.Module): 
    def __init__(
            self, 
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            with_time_emb=True,
            schedule_opt=None,
            sample_proc = 'ddim',
            local_ensemble=True, 
            feat_unfold=True, 
            cell_decode=True,
            ppg_input_channels=3,
    ):
        super().__init__()
        self.denoiser1= GaussianDiffusionx2(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            with_time_emb=with_time_emb,
            schedule_opt=schedule_opt,
            sample_proc = sample_proc
        )
        self.denoiser2 = GaussianDiffusionx2(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            with_time_emb=with_time_emb,
            schedule_opt=schedule_opt,
            sample_proc = sample_proc
        )
        self.init_predictor = DFTHL1()
      
        self.cfc = CFC(dim=int(3), num_heads=1)
        
    def forward(self, condition,gt):

        dwt,idwt= DWT(),IWT()
        
        if self.training and USE_GRAD_CHECKPOINTING:
            x_,AA,HH = checkpoint(self.init_predictor, condition, use_reentrant=False)
        else:
            x_,AA,HH = self.init_predictor(condition)
        input_img = x_[:, :3, :, :]
        n, c, h, w = input_img.shape
        
       
        input_dwt = dwt(x_)
        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        x_HH,x_LL=self.cfc(input_LL, input_high0)

        if gt is not None :
            gt_dwt = dwt(gt)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]
            residual_LL = gt_LL - input_LL
            residual_high0 = gt_high0 - input_high0     
        else :
            residual_LL = None
            residual_high0 = None

        if self.training and USE_GRAD_CHECKPOINTING:
            noisell,x__LL = checkpoint(self.denoiser1, input_LL, residual_LL, x_LL, use_reentrant=False)
            noisehigh,x__high0 = checkpoint(self.denoiser2, input_high0, residual_high0, x_HH, use_reentrant=False)
        else:
            noisell,x__LL = self.denoiser1(input_LL,  residual_LL,x_LL)
            noisehigh,x__high0 = self.denoiser2(input_high0,residual_high0,x_HH)

        return x_, noisell,x__LL,noisehigh,x__high0,AA,HH


# ---------------------------------------------------------------- helpers around WfDiffx2 (not part of the repository's network)
_dwt, _idwt = DWT(), IWT()


@torch.no_grad()
def _wfdiff_infer_one(net, x):
    """WF-Diff inference for ONE image (x: 1x3xHxW), the same steps as WfdiffModel.test() in the repository:
    preliminary image x_ -> Haar DWT -> add the DDIM-sampled LL / high-frequency residuals -> inverse DWT."""
    out1, out2ll, _, out2high, _, _, _ = net(x, None)
    n = out1.shape[0]
    out1_dwt = _dwt(out1)
    out1_LL, out1_high0 = out1_dwt[:n], out1_dwt[n:]
    return _idwt(torch.cat((out1_LL + out2ll, out1_high0 + out2high), dim=0))


@torch.no_grad()
def wfdiff_infer(net, x):
    """x: BxCxHxW -> enhanced BxCxHxW. Images are processed one at a time on purpose: the repository's sampler returns
    only the last element of the batch dimension (`ret_img[-1]`), so a batched call would mix the images (see Section 1)."""
    return torch.cat([_wfdiff_infer_one(net, x[i:i + 1]) for i in range(x.size(0))], dim=0)


class WFDiffInference(nn.Module):
    """x -> enhanced image. Only used for MACs profiling (thop needs a plain forward(x))."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return wfdiff_infer(self.net, x)


