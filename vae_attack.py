import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf_F

from net import VAE
from tqdm import tqdm
from os.path import join
from argparse import Namespace
from torchvision import transforms
from torch.utils.data import DataLoader
from constant import IMPLEMENTED_LOSS


@torch.no_grad()
def attack(vae: VAE,
           loss_fn: nn.Module,
           x: torch.Tensor,
           eps: float = 16 / 255,
           alpha: float = 2 / 255,
           t: int = 10,
           v: [float, torch.Tensor] = None,
           reverse_direct: bool = False,
           progress_bar: bool = False):
    """
    :param vae: AutoencoderKL.
    :param loss_fn: Loss function to calculate the gradient.
    :param x: The image for attack.
    :param eps: The maximum changes n (of 255) for per pixel.
    :param alpha: The step size per optimization.
    :param t: The total iter of the attack.
    :param v: The target variance.
    :param reverse_direct: Change direct of grad.
    :param progress_bar: Show pbar or not.
    :return:  The adversarial sample.
    """
    cost_his = []
    delta = torch.zeros_like(x, device=x.device)
    x_adv = x + delta
    if loss_fn._get_name() != "mse" and v is None:
        v = torch.square(vae(x).std).detach()
        v = v.view(v.size(0), -1)
    pbar = tqdm(total=t, desc='attack', disable=not progress_bar)
    time_start = time.time()
    for i in range(t):
        x_adv.requires_grad_()
        with torch.enable_grad():
            posterior = vae(x_adv)
            if loss_fn._get_name() == "mse":
                cost = loss_fn(posterior)
            else:
                cost = loss_fn(posterior, v=v, reverse_direct=reverse_direct)
            cost.backward()
            sign_grad = x_adv.grad.data.sign()
        cost_his.append(cost.item())
        delta += alpha * sign_grad
        delta = torch.clamp(delta, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, min=0, max=1)
        pbar.update(1)
    time_diff = time.time() - time_start
    pbar.close()
    return {"x_adv": x_adv.detach(),
            "cost_his": cost_his,
            "runtime": time_diff}



class Attack:
    def __init__(self, args: Namespace):
        self.vae = VAE(vae_from=args.vae_from, norm_from_01=True).to(args.device).eval()
        self.args = args
        self.to_pil = transforms.ToPILImage()
        if self.args.loss_fn != 'KL':
            self.loss_fn = IMPLEMENTED_LOSS[self.args.loss_fn](reduction='mean')
        else:
            self.loss_fn = IMPLEMENTED_LOSS[self.args.loss_fn](reduction='mean', v=self.args.v)

    def save_img(self, tensor, path):
        self.to_pil(tensor[0]).save(path, lossless=True, quality=100)

    def attack_loop(self, loader: DataLoader):
        pbar = tqdm(total=len(loader), desc=f'VAE attack with "{self.args.loss_fn}"')
        time_log = []
        for idx, (data, _, _) in enumerate(loader):
            data = data.to(self.args.device)
            results = attack(vae=self.vae,
                                 loss_fn=self.loss_fn,
                                 x=data,
                                 eps=self.args.eps,
                                 alpha=self.args.alpha,
                                 t=self.args.t,
                                 v=self.args.v,
                                 reverse_direct=self.args.reverse_direct,
                                 progress_bar=False)
            x_adv = results['x_adv']
            self.save_img(x_adv, join(self.args.x_adv_dir, f'{idx}.jpg'))
            time_log.append(results['runtime'])
            pbar.update(1)
        pbar.close()
        with open(join(self.args.log_dir, 'attack_avg_time.txt'), mode='w+') as f:
            print(sum(time_log) / len(time_log), file=f)
