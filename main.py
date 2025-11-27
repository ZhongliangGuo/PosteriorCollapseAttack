import os
import torch
import random
import numpy as np
from os.path import join
from dataset import SubValSet
from vae_attack import Attack
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from constant import IMPLEMENTED_LOSS, X_ADV_FOLDER_NAME, ADV_RECONSTRUCT_FOLDER_NAME, CLEAN_RECONSTRUCT_FOLDER_NAME, \
    IMPLEMENTED_MODEL_ID


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args() -> Namespace:
    parser = ArgumentParser()
    # dataset related arguments
    parser.add_argument("--dataset_path", type=str, default=r"/workspace/datasets/imagenet_1k_attack/data")
    parser.add_argument("--dataset_label", type=str,
                        default=r"/workspace/datasets/imagenet_1k_attack/label_with_caption.csv")
    parser.add_argument("--img_size", type=int, default=512)
    # attack related arguments
    parser.add_argument("--vae_from", type=str, choices=IMPLEMENTED_MODEL_ID.keys(), default='sd15')
    parser.add_argument("--loss_fn", type=str, choices=IMPLEMENTED_LOSS.keys(), default='PosteriorCollapseLoss')
    parser.add_argument("--eps", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=2)
    parser.add_argument("--t", type=int, default=40, help="num of iter for PGD")
    # logs
    parser.add_argument("--log_dir", type=str, default=None)
    # optional
    parser.add_argument("--v", type=float, default=1e-8, help="the target variance")
    parser.add_argument("--reverse_direct", action="store_true")
    # reproducibility
    parser.add_argument("--random_seed", type=int, default=3407)
    arguments = parser.parse_args()
    if arguments.v == -1:
        arguments.v = None
        print("using adaptive variance.")
    if arguments.log_dir is None:
        subfolder = (f"vae-from-{arguments.vae_from}_eps-{arguments.eps}_alpha-{arguments.alpha}_t-{arguments.t}_"
                         f"v-{arguments.v}_loss-{arguments.loss_fn}")
        arguments.log_dir = join('logs', subfolder)
    arguments.x_adv_dir = join(arguments.log_dir, X_ADV_FOLDER_NAME)
    os.makedirs(arguments.log_dir, exist_ok=True)
    os.makedirs(arguments.x_adv_dir, exist_ok=True)

    with open(join(arguments.log_dir, 'args.txt'), mode='w+') as f:
        print(arguments, file=f)
    arguments.eps = arguments.eps / 255
    arguments.alpha = arguments.alpha / 255
    arguments.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return arguments


def main():
    args = get_args()
    setup_seed(args.random_seed)
    attacker = Attack(args)
    loader = DataLoader(
        dataset=SubValSet(label_path=args.dataset_label,
                          folder_path=args.dataset_path,
                          img_size=args.img_size),
        batch_size=1,
        shuffle=False)
    attacker.attack_loop(loader)


if __name__ == "__main__":
    main()
