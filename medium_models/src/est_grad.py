import torch
import numpy as np

SQRT_PI = np.sqrt(2 * np.pi)
BF16_FRACTION_UPPER_BOUND = 0.00781260 
BF16_FRACTION_LOWER_BOUND = -0.00781260 

# Improve value efficiency by 60% 
def bf16_clip_value_inplace(tensor):
    mask_pos = (tensor > 0) & (tensor < BF16_FRACTION_UPPER_BOUND)
    tensor[mask_pos] = BF16_FRACTION_UPPER_BOUND
    
    mask_neg = (tensor < 0) & (tensor > BF16_FRACTION_LOWER_BOUND)
    tensor[mask_neg] = BF16_FRACTION_LOWER_BOUND
    
    return tensor

#precompute log_scale
def log_scale_fn(std: float) -> float:
    log_scale =  float(np.log(std * SQRT_PI))
    # print(log_scale)
    return log_scale

# 2.441793441772461 -> 0.662970781326294 s
# def generate_log_prob(sample, mean, std=0.01)#, log_scale=-3.6862316527834187):
def generate_log_prob(sample, mean, std: float):
    var = std ** 2
    log_scale = float(np.log(std * SQRT_PI))
    diff_squared = (sample - mean).pow(2)
    return -(diff_squared / (2 * var)) - log_scale


## 5.383518934249878 -> 0.019884109497070312 s
# def perturbation(x_3d, std, stop_ind=1, log_scale=-3.6862316527834187):
def perturbation(x_3d, std: float):
    n_sentence, n_word, d_word = x_3d.shape
    n = n_sentence * n_word
    n_half = n // 2
    x = x_3d.view(-1, d_word)
    x_noisy = x.clone()
    noise = torch.randn(n_half, d_word, device=x.device, dtype=torch.float32) * std
    noise = bf16_clip_value_inplace(noise).to(x.dtype)
    
    # 15 ~ 20% speedup
    x_noisy[:n_half].add_(noise)
    x_noisy[n_half:n].sub_(noise)
    x_noisy = x_noisy.detach()

    log_prob = torch.mean(generate_log_prob(x_noisy, x, std=std).view(n,-1), dim=1) 
    return x_noisy.view(n_sentence, n_word, d_word), log_prob