import torch
import torch.nn as nn
from .DNeuro import *
from timm.layers import trunc_normal_
from spikingjelly.activation_based import neuron, layer



def Linear(in_planes, out_planes, bias=False):
    return layer.Linear(in_planes, out_planes, bias=bias)


def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return layer.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return layer.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


def NeuroNode(v_threshold=1.):
    return DLIFNode(detach_reset=True, decay_input=True, v_threshold=v_threshold, D_threshold=-0.05)
    # return LIFNode(detach_reset=True, decay_input=True, v_threshold=v_threshold)
    # return neuron.LIFNode(detach_reset=True, decay_input=True, v_threshold=v_threshold)
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_sn = NeuroNode()
        self.fc1_conv = Conv2d(in_features, hidden_features, kernel_size=1)
        self.fc1_bn = layer.BatchNorm2d(hidden_features)

        self.fc2_sn = NeuroNode()
        self.fc2_conv = Conv2d(hidden_features, out_features, kernel_size=1)
        self.fc2_bn = layer.BatchNorm2d(out_features)

    def forward(self, x):                                       # T, B, C, H, W
        identity = x

        x = self.fc1_sn(x)                                      
        x = self.fc1_conv(x)                                    # T, B, hidden_features, H, W
        x = self.fc1_bn(x)
        
        x = self.fc2_sn(x)                                      
        x = self.fc2_conv(x)                                    # T, B, C, H, W
        x = self.fc2_bn(x)

        x = x + identity
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads

        self.proj_sn = NeuroNode()
        self.q_conv = Conv1d(dim, dim, kernel_size=1)
        self.q_bn = layer.BatchNorm1d(dim)

        self.q_sn = NeuroNode()
        self.k_conv = Conv1d(dim, dim, kernel_size=1)
        self.k_bn = layer.BatchNorm1d(dim)

        self.k_sn = NeuroNode()
        self.v_conv = Conv1d(dim, dim, kernel_size=1)
        self.v_bn = layer.BatchNorm1d(dim)
        self.v_sn = NeuroNode()

        self.attn_sn = NeuroNode(0.5)
        self.proj_conv = Conv1d(dim, dim, kernel_size=1)
        self.proj_bn = layer.BatchNorm1d(dim)

    def forward(self, x):                                           # T, B, C, H, W
        T, B, C, H, W = x.shape
        N = H * W
        identity = x
        x_for_qkv = self.proj_sn(x).flatten(3)                      # T, B, C, N

        q_conv_out = self.q_conv(x_for_qkv)                         # T, B, C, N
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_sn(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()        # T, B, num_heads, N, C//num_heads

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_sn(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()        # T, B, num_heads, N, C//num_heads

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_sn(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()        # T, B, num_heads, N, C//num_heads

        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)                               # T, B, num_heads, 1, C//num_heads
        kv = self.attn_sn(kv)
        x = q.mul(kv)                                                   # T, B, num_heads, N, C//num_heads

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()          # T, B, C, N
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W).contiguous()

        x = x + identity
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class MS_SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, imagenet=True):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size)
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.imagenet = imagenet

        self.proj_conv0 = Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn0 = layer.BatchNorm2d(embed_dims//8)

        self.proj_sn1 = NeuroNode()
        self.proj_conv1 = Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = layer.BatchNorm2d(embed_dims//4)

        self.proj_sn2 = NeuroNode()
        self.proj_conv2 = Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = layer.BatchNorm2d(embed_dims//2)

        self.proj_sn3 = NeuroNode()
        self.proj_conv3 = Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = layer.BatchNorm2d(embed_dims)

        self.proj_sn4 = NeuroNode()
        self.proj_conv4 = Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn4 = layer.BatchNorm2d(embed_dims)

        if imagenet:
            self.block1_mp = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            self.block2_mp = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_mp = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_mp = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    
    def forward(self, x):
        x = self.proj_conv0(x)
        x = self.proj_bn0(x)

        x = self.proj_sn1(x)
        x = self.block1_mp(x) if self.imagenet else x
        x = self.proj_conv1(x)
        x = self.proj_bn1(x)

        x = self.proj_sn2(x)
        x = self.block2_mp(x) if self.imagenet else x
        x = self.proj_conv2(x)
        x = self.proj_bn2(x)

        x = self.proj_sn3(x)
        x = self.block3_mp(x)
        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.block4_mp(x)

        x_feat = x
        x = self.proj_sn4(x)
        x = self.proj_conv4(x)
        x = self.proj_bn4(x)
        x = x + x_feat

        return x


class SpikeDrivenTransformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, imagenet=True
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.patch_embed = MS_SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims,
                                 imagenet=imagenet)

        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        # classification head
        self.head_lif = NeuroNode()
        self.head = nn.Linear(embed_dims, num_classes)

        for m in self.modules():
            if isinstance(m, (layer.Linear, nn.Linear, layer.Conv1d, layer.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (layer.BatchNorm1d, layer.BatchNorm2d)):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)             # T, B, 3, H, W

        x = self.patch_embed(x)                                     # T, B, embed_dims, H, W
        for blk in self.block:
            x = blk(x)                                              # T, B, embed_dims, H, W
        x = x.flatten(3).mean(3)                                    # T, B, embed_dims
        x = self.head_lif(x)                                        # T, B, embed_dims
        x = self.head(x.mean(0))                                    # B, num_classes
        return x


def spikedriventransformer_cifar(num_classes=10, T=4, depths=4, num_heads=12, embed_dims=384):
    model = SpikeDrivenTransformer(
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=4,
        in_channels=3, num_classes=num_classes, qkv_bias=False,
        depths=depths, sr_ratios=1, T=T, imagenet=False
    )
    return model


def spikedriventransformer_imagenet(num_classes=1000, T=4, depths=8, num_heads=8, embed_dims=384):
    model = SpikeDrivenTransformer(
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=4,
        in_channels=3, num_classes=num_classes, qkv_bias=False,
        depths=depths, sr_ratios=1, T=T, imagenet=True
    )
    return model


def spikedriventransformer(imagenet=True, **kwargs):
    if imagenet:
        return spikedriventransformer_imagenet(**kwargs)
    else:
        return spikedriventransformer_cifar(**kwargs)