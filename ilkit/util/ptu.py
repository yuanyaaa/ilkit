"""Pytorch Utils, namely, operations associated with Tensor and Network"""

from typing import Iterable, List, Tuple, Union

import torch as th

# from GPUtil import showUtilization as gpu_usage
from numba import cuda
from torch import nn
from torch.optim import Optimizer

# --------------------- Setting --------------------


def set_torch():
    th.set_default_dtype(th.float32)
    th.utils.backcompat.broadcast_warning.enabled = True
    th.utils.backcompat.keepdim_warning.enabled = True


def clean_cuda():
    th.cuda.empty_cache()
    for gpu_id in range(th.cuda.device_count()):
        cuda.select_device(gpu_id)
        cuda.close()


# --------------------- Tensor ---------------------


def tensor2ndarray(tensors: Tuple[th.Tensor]):
    """Convert torch.Tensor to numpy.ndarray
    """
    result = []
    for item in tensors:
        if th.is_tensor(item):
            result.append(item.detach().cpu().numpy())
        else:
            result.append(item)
    return result


# ------------------- Manipulate NN Module ----------------------


def move_device(modules: List[th.nn.Module], device: Union[str, th.device]):
    """Move net to specified device
    """
    for module in modules:
        module.to(device)


def freeze_net(nets: List[nn.Module]):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = False


# ------------------ Initialization ----------------------------
def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# ------------------ Regularization -----------------------------


def orthogonal_reg(model: nn.Module, device: Union[th.device, str], beta: float = 1e-4):
    """Orthogonal regularization
    
    Paper: 
    [1] https://arxiv.org/pdf/1609.07093.pdf
    
    Code:
    [1] https://github.com/kevinzakka/pytorch-goodies
    [2] https://zhuanlan.zhihu.com/p/98873800
    """
    reg = 1e-6
    orth_loss = 0.0
    for name, param in model.named_parameters():
        if "bias" not in name:
            param_flat = param.view(param.shape[0], -1)
            sym = th.mm(param_flat, th.t(param_flat))
            sym -= th.eye(param_flat.shape[0]).to(device)
            orth_loss += reg * sym.abs().sum()
    return beta * orth_loss


# ----------------------- Optimization ----------------------------


def gradient_descent(
    net_optim: Optimizer,
    loss: th.Tensor,
    parameters: Union[th.Tensor, Iterable[th.Tensor]] = None,
    max_grad_norm: float = None,
    retain_graph: bool = False,
):
    """Update network parameters with gradient descent.
    """
    net_optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clip
    if all([parameters, max_grad_norm]):
        th.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

    net_optim.step()
    return loss.item()
