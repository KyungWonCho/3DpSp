import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inversion.models.helpers import EqualLinear, get_blocks, bottleneck_IR, bottleneck_IR_SE

'''
This code is adapted from pixel2style2pixel (https://github.com/eladrich/pixel2style2pixel)
'''


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial, camera):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.camera = camera
        if camera:
            self.in_c = in_c + 16   # feature channel in_c + camera channel 16
        else:
            self.in_c = in_c
        self.spatial = spatial
        if camera:
            # linear mapping for camera parameters: 25 to 16
            self.linear_camera = nn.Sequential(
                nn.Linear(25, 16, bias=True),
                nn.LeakyReLU(),
                nn.Linear(16, 16, bias=True),
                nn.Sigmoid()
            )
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            nn.Conv2d(self.in_c, self.in_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        ]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(self.out_c, self.out_c, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(self.out_c, self.out_c, lr_mul=1)

    def forward(self, x, p):
        if self.camera:
            assert p is not None, "camera parameter is required"
            p = self.linear_camera(p)
            p = p.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.spatial, self.spatial)
            # x: [B, 512, spatial, spatial] -> [B, 528, spatial, spatial]
            x = torch.cat([x, p], dim=1)
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(nn.Module):
    def __init__(self, num_layers=50, mode='ir_se', camera=True):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            # for ReStyle 3 + 3 = 6
            nn.Conv2d(6, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules) # backbone
        self.styles = nn.ModuleList()
        for i in range(14):
            if i < 3:       # coarse
                style = GradualStyleBlock(512, 512, 16, camera)
            elif i < 7:     # intermediate
                style = GradualStyleBlock(512, 512, 32, camera)
            else:           # fine
                style = GradualStyleBlock(512, 512, 64, camera)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, p=None):
        # x: [B, 6, 256, 256] to [B, 64, 256, 256]
        x = self.input_layer(x)
        
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
                
        # c3: [B x 512 x 16 x 16]
        for j in range(3):
            # [B x 512 x 16 x 16] + [B x 25] -> [B x 512] (W)
            latents.append(self.styles[j](c3, p))
        
        # latlayer1 c2: [256 x 32 x 32] -> [512 x 32 x 32] (pixel-wise Linear)
        # _upsample_add c3: [512 x 16 x 16] -> [512 x 32 x 32] upsample, add.
        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(3, 7):
            # [B x 512 x 32 x 32] + [B x 25] -> [B x 512] (W)
            latents.append(self.styles[j](p2, p))
        
        # latlayer2 c1: [128 x 64 x 64] -> [256 x 32 x 32] (pixel-wise Linear)
        # _upsample_add p2: [512 x 64 x 64] -> [512 x 64 x 64] upsample, add.
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(7, 14):
            # [B x 512 x 64 x 64] + [B x 25] -> [B x 512] (W)
            latents.append(self.styles[j](p1, p))

        # out: [B x 14 x 512]
        out = torch.stack(latents, dim=1)
        return out
