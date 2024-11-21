import os
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from inversion.datasets.dataset import ImageFolderDataset
from inversion.models.psp import pSp
from inversion.options.test_options import TestOptions
from inversion.training.utils import tensor2im, synthesis_avg_image


def main():
    test_opts = TestOptions().parse()

    out_path = os.path.join(test_opts.exp_dir, 'results')
    os.makedirs(out_path, exist_ok=True)

    dataset = ImageFolderDataset(path=test_opts.data_path, resolution=None, load_conf_map=True, use_labels=True)
    dataloader = DataLoader(dataset)

    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = pSp(opts).cuda().eval()

    assert "state_dict" in ckpt, "checkpoint don't have pSp encoders"
    net.encoder.load_state_dict(ckpt["state_dict"])
    assert "latent_avg" in ckpt, "checkpoint don't have latent average"
    latent_avg = ckpt["latent_avg"].cuda()
    net.latent_avg = latent_avg
    avg_image = synthesis_avg_image(opts.device, net.decoder, net.latent_avg)

    for x_resized, _, camera_parameter, _, _, _, fname in tqdm(dataloader):
        x_resized, camera_parameter = x_resized.cuda().float(), camera_parameter.cuda().float()
        with torch.no_grad():
            latent = None
            for iter in range(opts.n_iters_per_batch):
                if iter == 0:
                    avg_image_for_batch = avg_image.unsqueeze(0).repeat(x_resized.shape[0], 1, 1, 1)
                    x_input = torch.cat([x_resized, avg_image_for_batch], dim=1)
                else:
                    x_input = torch.cat([x_resized, y_hat_resized], dim=1)
                latent, y_hat, _ = net.forward(x_input, camera_parameter.detach(), None, latent)
                y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))
        for i in range(y_hat.shape[0]):
            result = tensor2im(y_hat[i])
            im_save_path = os.path.join(out_path, f'{fname[i]}.png')
            result.save(im_save_path)


if __name__ == '__main__':
    main()
