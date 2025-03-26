import numpy as np
from dataset.NEUdata import get_loader_brats
from denoising_diffusion_pytorch.simple_diffusion import *
import torch
import torch.nn as nn
import os
from light_training.trainer import Trainer
from monai.utils import set_determinism
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
import torch.nn.functional as F
from PIL import Image
from model.net import net
import cv2

set_determinism(123)
test_dataset = ""
ddim_step = 10
exp = 'test'
uncer_step = 1
img_size = 352


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = net(mask_chans=1)

        betas = get_named_beta_schedule("cosine", 1000)

        self.snr_diffusion = GaussianDiffusion(image_size=img_size,
                                               channels=1,
                                               model=self.model,
                                               noise_d=64,
                                               num_sample_steps=ddim_step,
                                               clip_sample_denoised=True,
                                               pred_objective='x0')

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, note=0):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            times = torch.zeros((x.shape[0],), device=x.device).float().uniform_(0, 1)
            x, log_snr = self.snr_diffusion.q_sample(x_start=x, times=times, noise=noise)
            return x, log_snr, noise

        elif pred_type == "denoise":
            return self.model(x, timesteps=step, cond_img=image)

        elif pred_type == "ddim_sample":
            sample_outputs = []
            for i in range(uncer_step):
                sample_out = self.snr_diffusion.sample((image.shape[0], 1, img_size, img_size), image, note=note)
                sample_outputs.append(sample_out)
            sample_return = torch.zeros((1, 1, img_size, img_size))
            for index in range(len(sample_out)):
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()
                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))
                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i][index].cpu()

            return sample_return


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, load_checkpoint=None,
                 val_start=None, num_gpus=1, logdir="./logs/test",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, load_checkpoint, val_start, device, val_every, num_gpus,
                         logdir, master_ip, master_port,
                         training_script)
        self.model = DiffUNet()
        self.count05 = self.count1 = self.count2 = 0

    def get_input(self, batch):
        image = batch[0]
        label = batch[1]
        name = batch[2]

        label = label.float()
        return image, label, name

    def save_img_cv2(self, res, name, type):
        res = res.squeeze()
        path = rf"E:\hr\work\Diff-UNet-main\BraTS2020\test_results\{test_dataset}\{exp}\{type}"
        os.makedirs(path, exist_ok=True)
        name = name.replace('.bmp', "")
        path = path + f"\\{name}.png"
        cv2.imwrite(path, res * 255)

    def validation_step(self, batch, thresh=None):
        image, label, name = self.get_input(batch)
        name = name[0].split('\\')[-1]
        pred = self.model(image, note=None, pred_type="ddim_sample")
        label_ = label.float().cpu().squeeze().numpy()
        o1 = F.interpolate(pred, size=label_.shape, mode='bilinear', align_corners=False)
        o1 = torch.sigmoid(o1)
        o1 = (o1 > 0.7).float().numpy().astype('uint8')
        self.save_img_cv2(o1, name, type="pred_cv2")
        o2 = o1.squeeze()
        wt = np.sum(np.abs(o2 - label_)) * 1.0 / (label_.shape[0] * label_.shape[1])
        return wt


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)

    return uncer_out


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_loader_brats()

    trainer = BraTSTrainer(env_type="pytorch",
                           max_epochs=300,
                           batch_size=8,
                           device="cuda:0",
                           val_every=1,
                           num_gpus=1,
                           master_port=17751,
                           training_script=__file__)

    logdir = r"best_model.pt"
    trainer.load_state_dict(logdir)

    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds, batch_size=8, thresh=0.7)
    print(f"\nmae_mean is {v_mean}")
