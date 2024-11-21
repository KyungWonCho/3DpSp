import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inversion.criteria.id_loss import IDLoss
from inversion.datasets.dataset import ImageFolderDataset
from inversion.models.psp import pSp
from inversion.training.ranger import Ranger
from inversion.training.utils import tensor2im, vis_faces, synthesis_avg_image, aggregate_loss_dict
from training.camera_utils import LookAtPoseSampler, FOV_to_intrinsics

matplotlib.use('Agg')

'''
This code is adapted from ReStyle (https://github.com/yuval-alaluf/restyle-encoder) and TriPlaneNet (https://github.com/anantarb/triplanenet)
'''

class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:0')
        self.opts.device = self.device
        self.global_step = 0

        # Initialize network
        self.net = pSp(opts).to(self.device)

        # loss function
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.lpips_loss = LPIPS(net='alex').to(self.device).eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        self.id_loss = IDLoss().to(self.device).eval()
        for param in self.id_loss.parameters():
            param.requires_grad = False

        # Initialize optimizer
        self.optimizer = Ranger(self.net.encoder.parameters(), lr=self.opts.learning_rate)

        # Initialize datasets
        self.train_dataset, self.test_dataset = self.configure_dataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opts.batch_size,
                                           num_workers=self.opts.num_workers, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.opts.batch_size,
                                          num_workers=self.opts.num_workers, shuffle=False, drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None

        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')

            if "state_dict" in ckpt:
                self.net.encoder.load_state_dict(ckpt["state_dict"])
                print("Load pSp encoder from checkpoint")

            if "step" in ckpt:
                self.global_step = ckpt["step"]
                print(f"Resuming training process from step {self.global_step}")

            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                print("Load optimizer from checkpoint")

            if "best_val_loss" in ckpt:
                self.best_val_loss = ckpt["best_val_loss"]
                print(f"Current best val loss: {self.best_val_loss}")

            if "latent_avg" in ckpt:
                self.net.latent_avg = ckpt["latent_avg"]
                print("Load latent average from checkpoint")

        if self.net.latent_avg is None:
            self.net.latent_avg = self.mean_latent(100000)
        self.avg_image = synthesis_avg_image(self.device, self.net.decoder, self.net.latent_avg).float().detach()

    # input [x_resized, x, x_mirror, camera_params, camera_params_mirror, conf_map_mirror]
    # y_0 = avg_image, (x_resized, y_0, cameras) : y_1 -> (x, y_1, cameras) : y_2 -> ... -> (x, y_4, cameras) -> y_5 (final)
    def perform_train_iteration_on_batch(self, x_resized, x, x_mirror, camera_params, camera_params_mirror,
                                         conf_map_mirror):
        loss_dict, latent, y_hat = None, None, None
        for iter in range(self.opts.n_iters_per_batch):
            if iter == 0:
                # avg_image_for_batch: [B x 3 x 256 x 256]
                avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x_resized.shape[0], 1, 1, 1)
                # x_input: [B x 6 x 256 x 256]
                x_input = torch.cat([x_resized, avg_image_for_batch], dim=1)
                # latent: [B x 14 x 512], y_hat: [B x 3 x 512 x 512], y_hat_mirror: [B x 3 x 512 x 512]
                latent, y_hat, y_hat_mirror = self.net.forward(x_input, camera_params.clone().detach(),
                                                               camera_params_mirror.clone().detach(), latent=None)
            else:
                # y_hat_clone: [B x 3 x 256 x 256]
                y_hat_clone = F.adaptive_avg_pool2d(y_hat, (256, 256)).detach().requires_grad_(True)
                latent_clone = latent.clone().detach().requires_grad_(True)
                # x_input: [B x 6 x 256 x 256]
                x_input = torch.cat([x_resized, y_hat_clone], dim=1)
                # latent: [B x 14 x 512], y_hat: [B x 3 x 512 x 512], y_hat_mirror: [B x 3 x 512 x 512]
                latent, y_hat, y_hat_mirror = self.net.forward(x_input, camera_params.clone().detach(),
                                                               camera_params_mirror.clone().detach(), latent_clone)

            loss, loss_dict = self.calc_loss(x, x_mirror, y_hat, y_hat_mirror, conf_map_mirror)

            loss.backward()

        return loss_dict, y_hat

    def train(self):
        self.net.encoder.train()
        torch.cuda.empty_cache()
        while self.global_step < self.opts.max_steps:
            for batch in tqdm(self.train_dataloader, desc=f"Training Step {self.global_step}"):
                self.global_step += 1

                self.optimizer.zero_grad(set_to_none=True)
                x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror, _ = batch
                x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror = x_resized.to(
                    self.device).float(), x.to(self.device).float(), camera_params.to(self.device).float(), x_mirror.to(
                    self.device).float(), camera_params_mirror.to(self.device).float(), conf_maps_mirror.to(
                    self.device).float()

                loss_dict, y_hat = self.perform_train_iteration_on_batch(x_resized, x, x_mirror, camera_params,
                                                                         camera_params_mirror, conf_maps_mirror)

                self.optimizer.step()

                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(x, y_hat, title='images/train')

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss:
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

    def perform_val_iteration_on_batch(self, x_resized, x, x_mirror, camera_params, camera_params_mirror,
                                       conf_map_mirror):
        loss_dict, latent, y_hat = None, None, None
        for iter in range(self.opts.n_iters_per_batch):
            if iter == 0:
                avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x_resized.shape[0], 1, 1, 1)
                x_input = torch.cat([x_resized, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([x_resized, y_hat], dim=1)

            latent, y_hat, y_hat_mirror = self.net.forward(x_input, camera_params, camera_params_mirror, latent=latent)

            _, loss_dict = self.calc_loss(x, x_mirror, y_hat, y_hat_mirror, conf_map_mirror)

            y_hat = F.adaptive_avg_pool2d(y_hat, (256, 256))

        return loss_dict, y_hat

    def validate(self):
        self.net.encoder.eval()

        agg_loss_dict = []
        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
            x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror, _ = batch
            x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror = x_resized.to(
                self.device).float(), x.to(self.device).float(), camera_params.to(self.device).float(), x_mirror.to(
                self.device).float(), camera_params_mirror.to(self.device).float(), conf_maps_mirror.to(
                self.device).float()
            with torch.no_grad():
                cur_loss_dict, y_hat = self.perform_val_iteration_on_batch(x_resized, x, x_mirror, camera_params,
                                                                           camera_params_mirror, conf_maps_mirror)
            agg_loss_dict.append(cur_loss_dict)

            self.parse_and_log_images(x, y_hat, title='images/test', subscript='{:04d}'.format(batch_idx))

        loss_dict = aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.encoder.train()
        return loss_dict

    def calc_loss(self, x, x_mirror, y_hat, y_hat_mirror, conf_map_mirror):
        loss_dict = {}

        loss_l2 = self.mse_loss(y_hat, x)
        loss_dict['loss_l2'] = float(loss_l2)
        loss = loss_l2 * self.opts.l2_lambda

        loss_id = self.id_loss(y_hat, x)
        loss_dict['loss_id'] = float(loss_id)
        loss += loss_id * self.opts.id_lambda

        loss_lpips = self.lpips_loss(y_hat, x).mean()
        loss_dict['loss_lpips'] = float(loss_lpips)
        loss += loss_lpips * self.opts.lpips_lambda

        loss_l2_mirror = torch.square(x_mirror - y_hat_mirror)
        loss_l2_mirror = loss_l2_mirror.mean(dim=1)
        loss_l2_mirror = loss_l2_mirror / (conf_map_mirror + 1)
        loss_l2_mirror = loss_l2_mirror.mean()
        loss_dict['loss_l2_mirror'] = float(loss_l2_mirror)
        loss += loss_l2_mirror * self.opts.l2_lambda_mirror

        loss_id_mirror = self.id_loss(y_hat_mirror, x_mirror)
        loss_dict['loss_id_mirror'] = float(loss_id_mirror)
        loss += loss_id_mirror * self.opts.id_lambda_mirror

        loss_lpips_mirror = self.lpips_loss(y_hat, x_mirror).mean()
        loss_dict['loss_lpips_mirror'] = float(loss_lpips_mirror)
        loss += loss_lpips_mirror * self.opts.lpips_lambda_mirror

        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, x, y_hat, title, subscript=None, display_count=4):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': tensor2im(x[i]),
                'y_hat': tensor2im(y_hat[i])
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None):
        fig = vis_faces(im_data)
        step = self.global_step
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.encoder.state_dict(),
            'opts': vars(self.opts),
            'best_val_loss': self.best_val_loss,
            'step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'latent_avg': self.net.latent_avg
        }
        return save_dict

    def configure_dataset(self):
        train_dataset = ImageFolderDataset(path=self.opts.train_dataset_path, resolution=None, load_conf_map=True,
                                           use_labels=True)
        test_dataset = ImageFolderDataset(path=self.opts.test_dataset_path, resolution=None, load_conf_map=True,
                                          use_labels=True)
        print(f'Number of training samples: {len(train_dataset)}')
        print(f'Number of test samples: {len(test_dataset)}')
        return train_dataset, test_dataset

    def mean_latent(self, n_latent):
        z_in = torch.from_numpy(np.random.randn(n_latent, self.net.decoder.z_dim)).float().to(self.device)
        cam_pivot = torch.tensor(self.net.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]),
                                 device=self.device)
        cam_radius = self.net.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
        intrinsic = FOV_to_intrinsics(fov_degrees=18.837, device=self.device).reshape(-1, 9).repeat(n_latent, 1)
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, batch_size=n_latent, device=self.device)
        cam2world_pose = cam2world_pose.reshape(-1, 16)
        camera_param = torch.cat((cam2world_pose, intrinsic), dim=1)
        w_mean = self.net.decoder.mapping(z_in, camera_param).mean(0, keepdim=True)
        return w_mean
