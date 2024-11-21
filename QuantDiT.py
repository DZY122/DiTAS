
"""
Quantization and optimization process of DiTAS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from torch.autograd import Variable
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
# from utils_qaunt_train import weight_quant_fn
from utils_qaunt import weight_quant_fn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from transformers import get_linear_schedule_with_warmup
import logging
import os
from glob import glob
from time import time
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
global_act_bit = None
global_weight_bit = None
import config
from LinearQuant import *



def main(args):

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    #torch.set_grad_enabled(False)
    torch.set_grad_enabled(True)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
######################################################################################################################

    torch.manual_seed(args.seed)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279, 27, 336, 177, 594, 748, 718, 447, 613]
    # class_labels = [153, 482, 629, 815, 234, 901, 436, 307, 62, 459, 192, 678, 837, 726, 518, 682]
    # class_labels = [214, 375, 492, 861, 134, 956, 421, 308, 56, 487, 201, 632, 794, 729, 510, 684]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    #model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # for DiTAS, the cfg_scale is set to be 1.5
    # for generation example, the cfg_scale is set to be 4
    model_kwargs = dict(y=y, cfg_scale=1.5)   
    torch.manual_seed(args.seed)


######################################################################################################################
######################################################################################################################
#                               Advanced weight quantization with traning-free LoRA                                  #
    print("Start to deploy Advanced weight quantization with traning-free LoRA......" )
    replace_module(model)
    model.eval()  # important!

######################################################################################################################
#                                           TAS (temporal-aggregated smoothing)                                      #
    print("Start to deploy TAS (temporal-aggregated smoothing)......" )
    torch.manual_seed(args.seed)
    set_find_scale(model)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    set_inference_optimize(model)

######################################################################################################################
#                                                 grid search optimization                                           #

    print("Start to deploy temporal grid search optimization......" )
    for i in range(112):
        activations.clear()
        torch.cuda.empty_cache()
        register_hooks(model, i)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        remove_hooks()
        optimization(model, i)


    with torch.no_grad():
        smooth(model)

    para(model)
    model.eval()  # important!
    set_inference(model)
    torch.set_grad_enabled(False)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    model.eval() 

    # Sample images:
    torch.manual_seed(args.seed)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    #model_kwargs = dict(y=y, cfg_scale=1.5)
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "W4A8_50_256.png", nrow=4, normalize=True, value_range=(-1, 1))
    checkpoint = {
            "model": model.state_dict()
        }
    checkpoint_path = "W4A8_50_256.pt"
    torch.save(checkpoint, checkpoint_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(torch.cuda.is_available())
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256) #256 to 128
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=1000)
    # for DiTAS, the cfg_scale is set to be 1.5
    # for generation example, the cfg_scale is set to be 4
    parser.add_argument("--cfg-scale", type=float, default=4.0) 
    parser.add_argument("--per-proc-batch-size", type=int, default=2)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--act-bit", type=int, default=8)
    parser.add_argument("--weight-bit", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()

    global_act_bit = args.act_bit
    global_weight_bit = args.weight_bit
    main(args)