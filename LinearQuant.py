"""
Quantization and optimization process of DiTAS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from config import global_act_bit
from config import global_weight_bit



def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger

class LinearQuantLoRA(nn.Module):
    def __init__(self, in_feature, out_feature, reduced_rank, has_bias=True, args=None):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.reduced_rank = reduced_rank
        self.has_bias = has_bias
        self.quant = nn.Linear(in_feature, out_feature, bias=False)
        self.ln = nn.Linear(in_feature, out_feature, bias=False)

        has_bias=True

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature, requires_grad=False))

        self.has_svd_adapter = True
        self.has_lora_adapter = False
        self.activate = True
        # self.scales = nn.Parameter(torch.ones(in_feature))
        self.scales = None
        self.merge = False

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.calibrate = True
        self.findscale = False
        self.optimize_scale = False
        self.inference = False
        self.inference_optimize = False
        self.activation_scale = None
        self.weight_scale = None

        self.activation_scale_max = None

        self.count = 1
        self.forwardcount = 1

        if self.has_svd_adapter:
            self.right = nn.Linear(in_feature, reduced_rank, bias=False)
            self.left = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_lora_adapter:
            self.lora_A = nn.Linear(in_feature,reduced_rank, bias=False)
            self.lora_B = nn.Linear(reduced_rank, out_feature, bias=False)


    def forward(self, x):

        HX = 0
        LRX = 0

        if self.activate == True:
            if self.calibrate:
                HX = self.quant(x)
                right_output = self.right(x) if self.has_svd_adapter else 0
                LRX = self.left(right_output) if self.has_svd_adapter else 0

            if self.findscale:
                activation_scale = x.view(-1, x.shape[-1]).abs().max(dim=0)[0].clamp_(min=1e-4)
                if self.activation_scale_max == None:
                    self.activation_scale_max = activation_scale
                else:
                    self.activation_scale_max = torch.max(activation_scale, self.activation_scale_max)
                HX = self.ln(x)
                LRX = 0
                
            if self.inference:
                if self.merge == False:
                    x = torch.div(x.detach(), self.scales)
                maxvalue = torch.max(x)
                minvalue = torch.min(x)
                x = weight_quant_fn(x,
                            num_bits=global_act_bit,
                            quant_method='asymmetric', activate = True, maxvalue = maxvalue, minvalue = minvalue)
                HX = self.quant(x)
                right_output = self.right(x) if self.has_svd_adapter else 0
                LRX = self.left(right_output) if self.has_svd_adapter else 0

            if self.inference_optimize:
                HX = self.ln(x)
                LRX = 0

        else:
            HX = self.quant(x)
            right_output = self.right(x) if self.has_svd_adapter else 0
            LRX = self.left(right_output) if self.has_svd_adapter else 0


        Y = HX + LRX + self.bias if self.has_bias else HX + LRX
        if self.has_lora_adapter:
            lora_A_output = self.lora_A(x)
            Y += self.lora_B(lora_A_output)

        Y = Y.detach()
        torch.cuda.empty_cache()
        return Y

    def initialize_weight(self, quant_weight, original_weight, left_weight, right_weight, sparse_weight=None, bias=None, scales=None):
        self.quant.weight = nn.Parameter(quant_weight, requires_grad=False)  # Freeze the backbone
        self.ln.weight = nn.Parameter(original_weight, requires_grad=False) 
        self.scales =scales

        self.has_bias=True

        if self.has_bias:
            self.bias = nn.Parameter(bias, requires_grad=True)
        if self.has_svd_adapter:
            self.left.weight = nn.Parameter(left_weight, requires_grad=True)
            self.right.weight = nn.Parameter(right_weight, requires_grad=True)

    def initialize_weight_sample(self, quant_weight, left_weight, right_weight, sparse_weight=None, bias=None):
        self.quant.weight = nn.Parameter(quant_weight, requires_grad=False)  # Freeze the backbone
        del self.ln
        self.scales = nn.Parameter(torch.ones(self.in_feature))

        self.has_bias=True

        if self.has_bias:
            self.bias = nn.Parameter(bias, requires_grad=True)
        if self.has_svd_adapter:
            self.left.weight = nn.Parameter(left_weight, requires_grad=True)
            self.right.weight = nn.Parameter(right_weight, requires_grad=True)




activations = []
hook_handles = []

layer_names = []

for block in range(28):
    base_name = f"blocks.{block}"
    layer_names.append(f"{base_name}.attn.qkv")
    layer_names.append(f"{base_name}.attn.proj")
    layer_names.append(f"{base_name}.mlp.fc1")
    layer_names.append(f"{base_name}.mlp.fc2")


# Print the result to verify
# for name in layer_names:
#     print(name)

def forward_hook(module, input, output):
    activations.append(input)

def register_hooks(module, number, parent_name=''):
    global hook_handles
    for name, submodule in module.named_children():
        allow_name = ['attn', 'mlp']
        full_name = parent_name + '.' + name if parent_name else name
        if full_name == layer_names[number]:
            hook_handle = submodule.register_forward_hook(forward_hook)
            hook_handles.append(hook_handle)
            print(f"Hook registered for layer: {full_name}")
        register_hooks(submodule, number, full_name)


def remove_hooks():
    global hook_handles
    for hook_handle in hook_handles:
        hook_handle.remove()
    hook_handles = []


def optimize(model):
    for i in range(112):
        activations.clear()
        register_hooks(model, i)



def set_device(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            device = target_attr.quant.weight.device
            target_attr.scales = target_attr.scales.to(device)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        set_device(immediate_child_module)



def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param      rank_ratio: rank_of_decomposed_matrix
    :return: L, R
    """

    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    reduced_rank = int(reduced_rank)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R



def replace_module(module,
                             allow_name=None,
                             block_name=None,
                             reduced_rank=32,
                             decomposition=True,
                             quant_method='uniform',
                             int_bit=4,
                             args=None):
    """
    :param         int_bit: integer bit, 8, 4, 2 for example
    :param    quant_method: quantization method to use
    :param   decomposition: whether quantize
    :param          module: an nn.Module class
    :param      block_name: do not continue to iterate when the module's name is in the block_name
    :param      allow_name: replace the module if its name is in the allow_name
    :param reduced_rank: rank of low rank adapters
    :return: None
    """

    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['proj', 'qkv', 'fc1', 'fc2']

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == torch.nn.Linear and any(attr_str in an for an in allow_name):
            target_attr = getattr(module, attr_str)
            L, R = 0, 0
            for i in range(10):
                low_rank_product = L @ R if torch.is_tensor(L) else 0
                residual = target_attr.weight - low_rank_product
                quant_w= weight_quant_fn(residual,
                    num_bits=global_weight_bit,
                    quant_method='asymmetric')
                output = low_rank_decomposition(target_attr.weight  - quant_w, reduced_rank=reduced_rank)
                L, R = output[0], output[1]

            # target_attr.weight = torch.nn.Parameter(quant_w)
            linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                            has_bias=True,  args=args)
            linear_loras.initialize_weight(quant_w, target_attr.weight, L, R, 0, target_attr.bias)
            #linear_loras.initialize_weight(quant_w, 0, 0, 0, target_attr.bias)
            setattr(module, attr_str, linear_loras)
        
        elif type(target_attr) == LinearQuantLoRA and any(attr_str in an for an in allow_name):
            target_attr = getattr(module, attr_str)
            L, R = 0, 0
            for i in range(10):
                low_rank_product = L @ R if torch.is_tensor(L) else 0
                residual = target_attr.quant.weight - low_rank_product
                quant_w= weight_quant_fn(residual,
                    num_bits=global_weight_bit,
                    quant_method='asymmetric')
                output = low_rank_decomposition(target_attr.quant.weight  - quant_w, reduced_rank=reduced_rank)
                L, R = output[0], output[1]

            # target_attr.quant.weight = torch.nn.Parameter(quant_w)
            linear_loras = LinearQuantLoRA(target_attr.in_feature, target_attr.out_feature, reduced_rank=int(reduced_rank),
                                            has_bias=True,  args=args)
            linear_loras.initialize_weight(quant_w, target_attr.ln.weight, L, R, 0, target_attr.bias)
            #linear_loras.initialize_weight(quant_w, 0, 0, 0, target_attr.bias)
            setattr(module, attr_str, linear_loras)


    for name, immediate_child_module in module.named_children():
        replace_module(immediate_child_module)


def replace_module_sample(module,
                             device,
                             allow_name=None,
                             block_name=None,
                             reduced_rank=32,
                             decomposition=True,
                             quant_method='uniform',
                             int_bit=4,
                             args=None):
    """
    :param         int_bit: integer bit, 8, 4, 2 for example
    :param    quant_method: quantization method to use
    :param   decomposition: whether quantize
    :param          module: an nn.Module class
    :param      block_name: do not continue to iterate when the module's name is in the block_name
    :param      allow_name: replace the module if its name is in the allow_name
    :param reduced_rank: rank of low rank adapters
    :return: None
    """

    # Default allow name and block name lists
    allow_name = ['proj', 'qkv', 'fc1', 'fc2']
    allow_name1 = ['proj', 'fc2']
    allow_name2 = ['qkv', 'fc1']
    # if block_name is None:
    #     block_name = ['attn', 'mlp']

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == torch.nn.Linear and any(attr_str in an for an in allow_name):
            target_attr = getattr(module, attr_str)
            quant_w= weight_quant_fn(target_attr.weight,
                num_bits=global_weight_bit,
                quant_method='asymmetric')
            linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                            has_bias=True,  args=args)
            linear_loras.initialize_weight_sample(quant_w, torch.zeros_like(linear_loras.left.weight).to(device), torch.zeros_like(linear_loras.right.weight).to(device), 0, target_attr.bias)

            setattr(module, attr_str, linear_loras)

        elif type(target_attr) == LinearQuantLoRA and any(attr_str in an for an in allow_name):
            target_attr = getattr(module, attr_str)

            quant_w= weight_quant_fn(target_attr.quant.weight,
                num_bits=global_weight_bit,
                quant_method='asymmetric')

            target_attr.quant.weight = torch.nn.Parameter(quant_w)
            linear_loras = LinearQuantLoRA(target_attr.in_feature, target_attr.out_feature, reduced_rank=int(reduced_rank),
                                            has_bias=True,  args=args)
            linear_loras.initialize_weight_sample(quant_w, torch.zeros_like(linear_loras.left.weight).to(device), torch.zeros_like(linear_loras.right.weight).to(device), 0, target_attr.bias)

            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():

        replace_module_sample(immediate_child_module, device)



def initial_module(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            target_attr = getattr(module, attr_str)
            target_attr.maxvalue = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            target_attr.minvalue = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        initial_module(immediate_child_module)

def set_inference(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            target_attr = getattr(module, attr_str)
            target_attr.calibrate = False
            target_attr.findscale = False
            target_attr.inference = True
            target_attr.optimize_scale = False
            target_attr.inference_optimize = False

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        set_inference(immediate_child_module)


def set_inference_optimize(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            target_attr = getattr(module, attr_str)
            target_attr.calibrate = False
            target_attr.findscale = False
            target_attr.inference = False
            target_attr.inference_optimize = True
            target_attr.optimize_scale = False

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        set_inference_optimize(immediate_child_module)

def set_find_max_min(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            target_attr = getattr(module, attr_str)
            target_attr.calibrate = True
            target_attr.findscale = False
            target_attr.inference = False
            target_attr.optimize_scale = False
            target_attr.inference_optimize = False

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        set_find_max_min(immediate_child_module)

def set_find_scale(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            target_attr = getattr(module, attr_str)
            target_attr.calibrate = False
            target_attr.findscale = True
            target_attr.inference = False
            target_attr.optimize_scale = False
            target_attr.inference_optimize = False

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        set_find_scale(immediate_child_module)



def para(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == LinearQuantLoRA:
            target_attr.scales = nn.Parameter(target_attr.scales, requires_grad=True)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        # if any(name in bn for bn in block_name):
        #print(name)
        para(immediate_child_module)



#scale_name = ['qkv', 'fc1']
scale_name = ['proj', 'qkv', 'fc1', 'fc2']

parent_dict = {}
objects_dict = {}


def smooth(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        objects_dict[id(module)] = module
        objects_dict[id(target_attr)] = target_attr
        parent_dict[id(target_attr)] = id(module)

        if type(target_attr) == LinearQuantLoRA  and any(attr_str in an for an in scale_name):
            target_attr = getattr(module, attr_str)
            scale = target_attr.scales
            target_attr.quant.weight.mul_(scale.view(1, -1).detach())
            target_attr.right.weight.mul_(scale.view(1, -1).detach())
            del target_attr.ln
            

    for name, immediate_child_module in module.named_children():

        smooth(immediate_child_module)


def optimization(module, number, parent_name=''):
    global hook_handles
    for name, submodule in module.named_children():
        allow_name = ['proj', 'qkv', 'fc1', 'fc2']
        full_name = parent_name + '.' + name if parent_name else name
        if full_name == layer_names[number]:
            best_alpha = None
            best_loss = float("inf")
            loss_total = None
            for grid in range(21):
                print(best_loss)
                loss_total = 0
                alpha = 0.05 * grid
                for step in range(50):
                    with torch.no_grad():

                        input = activations[step][0]
                        torch.cuda.empty_cache()
                        torch.set_grad_enabled(True)
                        
                        submodule.activation_scale = submodule.activation_scale_max
                        submodule.activation_scale.requires_grad_(True)
                        device, dtype = submodule.quant.weight.device, submodule.quant.weight.dtype
                        submodule.weight_scale = submodule.quant.weight.abs().max(dim=0)[0].clamp(min=1e-4)

                        submodule.scales = (
                            ((submodule.activation_scale).pow(alpha)/ submodule.weight_scale.pow(1 - alpha))
                            .clamp(min=1e-4).view(-1)
                            .to(device)
                            .to(dtype)
                        )
                        submodule.scales.requires_grad_(True)

                        HX_original = None
                        original_quant_weight = submodule.quant.weight.detach().clone().to(device).to(dtype)
                        original_quant_weight_right = submodule.right.weight
                        original_quant_weight_left = submodule.left.weight


                        with torch.no_grad():
                            HX_original = submodule.ln(input) + submodule.bias

                        input_quant_1 = torch.div(input, submodule.scales)
                        input_quant_1.requires_grad_(True)
                        maxvalue = torch.max(input_quant_1)
                        minvalue = torch.min(input_quant_1)
                        input_quant = weight_quant_fn(input_quant_1, num_bits=global_act_bit, quant_method='asymmetric', activate = True, maxvalue = maxvalue, minvalue = minvalue)

                        # w_quant = weight_quant_fn(original_quant_weight, num_bits=global_weight_bit, quant_method='asymmetric')
                        w_right = original_quant_weight_right * submodule.scales.view(1, -1)

                        input_quant.requires_grad_(True)
                        # w_quant.requires_grad_(True)
                        HX_quant = F.linear(input_quant, original_quant_weight * submodule.scales.view(1, -1)) + F.linear(F.linear(input_quant, w_right), original_quant_weight_left) + submodule.bias

                        HX_quant.requires_grad_(True)


                        loss = mean_flat((HX_quant - HX_original) ** 2).mean().requires_grad_()
                        loss_total = loss_total + loss.item()

                print(loss_total)
                print(f'Best Loss: {best_loss}')
                if loss_total < best_loss:
                    best_loss = loss_total
                    best_alpha = alpha

                print("linear" + str(best_alpha))
            

            submodule.scales = (
                ((submodule.activation_scale).pow(best_alpha)/ submodule.weight_scale.pow(1 - best_alpha))
                .clamp(min=1e-4).view(-1)
                .to(device)
                .to(dtype)
            )

        optimization(submodule, number, full_name)


#merge the smoothing factor on activation to the previous layer's weight

scale_name_1 = ['qkv', 'fc1']
scale_name_2 = ['proj']

parent_dict = {}
objects_dict = {}

def merge(module):

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        objects_dict[id(module)] = module
        objects_dict[id(target_attr)] = target_attr
        parent_dict[id(target_attr)] = id(module)

        if type(target_attr) == LinearQuantLoRA  and any(attr_str in an for an in scale_name_1):
            target_attr = getattr(module, attr_str)
            scale = target_attr.scales
            target_attr.merge = True
            if attr_str == 'qkv':
                weight = objects_dict[parent_dict[id(module)]].adaLN_modulation[1].weight
                bias = objects_dict[parent_dict[id(module)]].adaLN_modulation[1].bias
                num_parts = 6
                part_size = weight.size(1) 
                part_weights = []
                part_biases = []
                for i in range(num_parts):
                    start_idx = i * part_size
                    end_idx = (i + 1) * part_size
                    part_weight = weight[start_idx:end_idx, :]
                    part_bias = bias[start_idx:end_idx]
                    part_biases.append(part_bias)
                    part_weights.append(part_weight)
                # print(scale.size())
                # print(scale)

                part_weights[0].div_(target_attr.scales.view(-1, 1).detach().to(weight.device))
                part_biases[0].div_(target_attr.scales.view(-1, 1).squeeze(1).detach().to(weight.device))
                part_weights[1].div_(target_attr.scales.view(-1, 1).detach().to(weight.device))
                part_biases[1].add_(1-target_attr.scales.view(-1, 1).squeeze(1).detach().to(weight.device)).div_(target_attr.scales.view(-1, 1).squeeze(1).detach().to(weight.device))

            else:
                weight = objects_dict[parent_dict[id(module)]].adaLN_modulation[1].weight
                bias = objects_dict[parent_dict[id(module)]].adaLN_modulation[1].bias
                num_parts = 6
                part_size = weight.size(1) 
                part_weights = []
                part_biases = []
                for i in range(num_parts):
                    start_idx = i * part_size
                    end_idx = (i + 1) * part_size
                    part_weight = weight[start_idx:end_idx, :]
                    part_bias = bias[start_idx:end_idx]
                    part_biases.append(part_bias)
                    part_weights.append(part_weight)

                part_weights[3].div_(target_attr.scales.view(-1, 1).detach().to(weight.device))
                part_biases[3].div_(target_attr.scales.view(-1, 1).squeeze(1).detach().to(weight.device))
                part_weights[4].div_(target_attr.scales.view(-1, 1).detach().to(weight.device))
                part_biases[4].add_(1-target_attr.scales.view(-1, 1).squeeze(1).detach().to(weight.device)).div_(target_attr.scales.view(-1, 1).squeeze(1).detach().to(weight.device))


        
        if type(target_attr) == LinearQuantLoRA and any(attr_str in an for an in scale_name_2):
            target_attr = getattr(module, attr_str)
            scale = target_attr.scales
            target_attr.merge = True

            weight_1 = module.qkv.quant.weight
            dim = int(weight_1.size(0)/3)
            v_weight = weight_1[2 * dim:3 * dim, :]
            v_weight.div_(scale.view(-1, 1).detach().to(v_weight.device))


            bias = module.qkv.bias
            dim = int(weight_1.size(0)/3)
            v_bias = bias[2 * dim:3 * dim]
            v_bias.div_(scale.view(-1, 1).squeeze(1).detach().to(v_weight.device))

            weight_2 = module.qkv.left.weight
            dim = int(weight_2.size(0)/3)
            left_weight_v_part = weight_2[2 * dim:3 * dim, :]
            left_weight_v_part.div_(scale.view(-1, 1).detach().to(v_weight.device))
            

    for name, immediate_child_module in module.named_children():

        merge(immediate_child_module)