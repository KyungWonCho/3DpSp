import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from training.camera_utils import FOV_to_intrinsics, LookAtPoseSampler


def tensor2im(var):
    var = var.detach().cpu().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 255] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(8, 4 * display_count))
    gs = fig.add_gridspec(display_count, 2)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        plt.imshow(hooks_dict['input_face'], cmap="gray")
        plt.title('Input')
        fig.add_subplot(gs[i, 1])
        plt.imshow(hooks_dict['y_hat'])
        plt.title('Output')
    plt.tight_layout()
    return fig


def synthesis_avg_image(device, network, latent_avg):
    intrinsic = FOV_to_intrinsics(18.837, device=device).reshape(-1, 9)
    cam_pivot = torch.tensor(network.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = network.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                              device=device).reshape(-1, 16)
    camera_param = torch.cat((cam2world_pose, intrinsic), dim=1)
    img_mean = network.synthesis(latent_avg, camera_param)['image'][0]
    img_mean = torch.nn.functional.interpolate(img_mean.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)
    return img_mean.squeeze(0)


def aggregate_loss_dict(agg_loss_dict):
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals
