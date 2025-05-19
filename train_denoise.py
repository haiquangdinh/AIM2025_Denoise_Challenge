import os

os.environ["OPENMP_NUM_THREADS"] = "16"

import numpy as np
import random
import time
import yaml
from tqdm import tqdm
import argparse
from accelerate import Accelerator, DistributedDataParallelKwargs

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from utils.utils import *

from models.ELD_models import UNetSeeInDark
from datasets.synth_train_dataset import SynthTrainDataset


def build_models(accelerator, args):
    model = UNetSeeInDark()
    return model


def build_dataloaders(accelerator, args):
    with open("./datasets/camera_config.yaml", "r") as f:
        cam_cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_set = SynthTrainDataset(
        clean_img_dir=args.clean_img_dir,
        benchmark_dir=args.benchmark_dir,
        camera_config=cam_cfg,
        iso_list=[800, 1600, 3200],
        dgain_range=args.train_dgain_range,
        patch_size=args.train_patch_size,
        inp_clip_low=False,
        inp_clip_high=True,
        n_crop_per_img=args.n_crop_per_img,
    )

    ## data loader
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.n_worker)
    accelerator.print(f">>>>> train_set: {len(train_set)}")
    return train_loader


def train_one_ep(model, train_loader, optimizer, criterion, accelerator, args):
    loss_am = AverageMeter("train_loss", ":.4e")
    psnr_am = AverageMeter("train_psnr", ":.2f")
    model.train()

    for data_id, data in enumerate(tqdm(train_loader)):
        clean, noisy = tensor_dim5to4(data["clean"]), tensor_dim5to4(data["noisy"])

        optimizer.zero_grad()
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        accelerator.backward(loss)
        optimizer.step()

        denoised, clean = accelerator.gather_for_metrics((denoised, clean))
        denoised, clean = torch.clamp(denoised, 0, 1), torch.clamp(clean, 0, 1)
        eval_res = psnr_ssim_metric_torch(denoised, clean)
        psnr_am.update(eval_res["psnr"])
        loss_am.update(loss.item())

    return loss_am.avg, psnr_am.avg


def main(args):
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    ## model
    model = build_models(accelerator, args)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch, eta_min=1e-5)
    criterion = nn.L1Loss()

    ## data init
    train_loader = build_dataloaders(accelerator, args)

    ## other init
    make_directory(f"./checkpoints/{args.task}")
    make_directory("./logs")
    logfile = f"./logs/{args.task}.log"

    ## resume training
    if args.resume_train is not None:
        cp = torch.load(args.resume_train, map_location="cpu", weights_only=False)
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        start_epoch = cp["epoch"] + 1
        accelerator.print(f">>>>> resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        ## logging argparse
        if accelerator.is_local_main_process:
            open(logfile, "w").close()
            with open(logfile, "w") as f:
                for arg, value in vars(args).items():
                    f.write(f"{arg}: {value}\n")
                f.write("\n\n\n\n\n")
        accelerator.print(">>>>> no resumed checkpoint, training from scratch")

    ## put all things to accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    ## train
    for epoch in range(start_epoch + 1, args.n_epoch + 1):
        train_loss, train_psnr = train_one_ep(model, train_loader, optimizer, criterion, accelerator, args)
        if not accelerator.optimizer_step_was_skipped:
            scheduler.step()

        ## checkpoint
        accelerator.wait_for_everyone()
        if epoch % args.save_freq == 0:
            state_dict = {
                "model": accelerator.get_state_dict(model),
                "optimizer": accelerator.get_state_dict(optimizer),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            accelerator.save(state_dict, f"./checkpoints/{args.task}/epoch_{epoch}.bin")
            time.sleep(5)

        ## log
        if accelerator.is_local_main_process:
            log(
                f"epoch: {epoch}, train loss: {train_loss:.4e}, train psnr: {train_psnr:.2f}",
                log=logfile,
                notime=True,
            )

    accelerator.end_training()


##--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## train-related
    parser.add_argument("--task", type=str, default="baseline")
    parser.add_argument("--resume_train", type=str, default=None)
    parser.add_argument("--n_epoch", type=int, default=1000)
    parser.add_argument("--train_patch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--n_crop_per_img", type=int, default=8)
    parser.add_argument("--train_dgain_range", type=list, default=[10, 200])
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--clean_img_dir", type=str, default="/datasets/sid_sony/Sony/long")
    parser.add_argument("--benchmark_dir", type=str, default="/datasets/aim_challenge_release_data")
    ## others
    parser.add_argument("--n_worker", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    _args = parser.parse_args(args=[])

    # fix seed
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    random.seed(_args.seed)
    torch.backends.cudnn.benchmark = True

    main(_args)
