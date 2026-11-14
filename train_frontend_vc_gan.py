"""Train the SoftVcStylizer frontend and full audio discriminator against a frozen DDSP model."""
import argparse

import torch
from torch.optim import lr_scheduler

from ddsp.checkpoint import load_ddsp_model
from discriminators.discriminator import Discriminator
from frontend import apply_overrides, options
from logger import utils
from optimizer.muon import Muon_AdamW
from reflow.data_loaders import get_data_loaders
from reflow.solver import train_vc_osgan
from reflow.vocoder import Vocoder
from stylizer.stylizer import SoftVcStylizer


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--base_checkpoint", default=None, help="DDSP/base reflow checkpoint or experiment directory")
    parser.add_argument("--frontend_checkpoint", default=None, help="Optional explicit frontend/GAN checkpoint to resume")
    return parser.parse_args(args=args, namespace=namespace)


def main():
    command = parse_args()
    args = utils.load_config(command.config)
    if command.device:
        args.device = command.device
    settings = apply_overrides(args, command)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.env.get("gpu_id", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.model.type != "RectifiedFlow":
        raise ValueError(f"Unknown base model type: {args.model.type}")
    if not args.env.get("style_dir"):
        raise ValueError("Set env.style_dir for frontend checkpoints and logs")
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    base_path = command.base_checkpoint or settings.get("base_checkpoint") or args.env.expdir
    model, base_args = load_ddsp_model(base_path, device=args.device)
    for key in ("sampling_rate", "block_size", "encoder_out_channels", "encoder"):
        if base_args.data.get(key) != args.data.get(key):
            raise ValueError(f"Base model and frontend config disagree on data.{key}")
    style_model = SoftVcStylizer(**(settings.get("model") or {})).to(args.device)
    model_d = Discriminator({}).to(args.device)
    optimizer = Muon_AdamW(style_model, muon_args={"weight_decay": args.train.weight_decay}, adamw_args={"weight_decay": 0})
    optimizer_d = Muon_AdamW(model_d, muon_args={"weight_decay": args.train.weight_decay}, adamw_args={"weight_decay": 0})
    step, style_model, optimizer, model_d, optimizer_d = utils.load_gan_model(
        settings.get("checkpoint") or args.env.style_dir, style_model, optimizer, model_d, optimizer_d, device=args.device,
        allow_missing_discriminator=True)
    for value in (optimizer, optimizer_d):
        for group in value.param_groups:
            group["initial_lr"] = args.train.lr
            group["lr"] = args.train.lr * args.train.gamma ** max((step - 2) // args.train.decay_step, 0)
            group["weight_decay"] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=step - 2)
    scheduler_d = lr_scheduler.StepLR(optimizer_d, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=step - 2)
    loader_train, loader_valid = get_data_loaders(args, frontend=True, load_audio=True,
            train_aug=False, aug_embd=False, use_short=True)
    train_vc_osgan(args, step, style_model, model, model_d, optimizer, optimizer_d,
                   scheduler, scheduler_d, vocoder, loader_train, loader_valid)


if __name__ == "__main__":
    main()
