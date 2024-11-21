import torch.nn as nn

import dnnlib
from inversion.configs.paths_config import model_paths
from inversion.models.psp_encoders import GradualStyleEncoder
from training import legacy

'''
This code is adapted from pixel2style2pixel (https://github.com/eladrich/pixel2style2pixel), ReStyle (https://github.com/yuval-alaluf/restyle-encoder) and TriPlaneNet (https://github.com/anantarb/triplanenet)
'''


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        self.encoder = GradualStyleEncoder(camera=opts.camera)
        with dnnlib.util.open_url(model_paths['eg3d']) as f:
            self.decoder = legacy.load_network_pkl(f)['G_ema']
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.latent_avg = None

    def forward(self, x, camera_params, novel_view_camera_params=None, latent=None):
        # x: [B, 6, 256, 256], camera_params: [B, 25], codes: [B, 14, 512] (residual of latent)
        codes = self.encoder(x, camera_params)

        if x.shape[1] == 6 and latent is not None:  # not first input
            codes = codes + latent
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        # y_hat: [B x 3 x 512 x 512]
        y_hat = self.decoder.synthesis(codes, camera_params, noise_mode='const')['image']
        if novel_view_camera_params is not None:
            # y_hat_novel: [B x 3 x 512 x 512]
            y_hat_novel = self.decoder.synthesis(codes, novel_view_camera_params, noise_mode='const')['image']
        else:
            y_hat_novel = None

        return codes, y_hat, y_hat_novel
