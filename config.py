"""
Configuration file for Latent Thought LM training.
Contains all hyperparameters and settings.
"""
import os
from datetime import datetime

MODEL_SIZE = "LTM-421M"
# MODEL_SIZE = "LTM-1.3B"

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
# DATA_CACHE_DIR = "data_owt"  # Directory for caching dataset
DATA_CACHE_DIR = "../Samba/data/slim"
batch_size = 64  # Micro-batch size (before gradient accumulation)
# max_seq_len = 1024  # Maximum sequence length
max_seq_len = 4096
# vocab_size = 50258  # Vocabulary size 
vocab_size = 32000

# -----------------------------------------------------------------------------
# 计算 TPI (Tokens Per Iteration)
# -----------------------------------------------------------------------------
# 匹配 Samba 的 global_batch_size = 512
# LTM 默认设置 gradient_accumulation_steps = 8, batch_size = 64
# 8 * 64 = 512，因此我们保留 LTM 的默认设置
gradient_accumulation_steps = 8
batch_size = 64  # Micro-batch size (per-GPU)

# TPI = Global Batch Size * Seq Len
TOKENS_PER_ITERATION = (gradient_accumulation_steps * batch_size) * max_seq_len
# TOKENS_PER_ITERATION = 512 * 4096 = 2,097,152

if MODEL_SIZE == "LTM-421M":
    # --- 架构 ---
    dim = 1536
    n_layers = 12
    n_heads = 12
    n_kv_heads = 12 # LTM 默认 n_kv_heads = n_heads
    
    # --- 训练 (20B Tokens) ---
    max_tokens = 20_000_000_000
    warmup_tokens = int(max_tokens * 0.01) # 1% warmup
    
    max_iters = max_tokens // TOKENS_PER_ITERATION         # 20B / 2.097M ≈ 9537
    warmup_iters = warmup_tokens // TOKENS_PER_ITERATION   # 0.2B / 2.097M ≈ 95
    lr_decay_iters = max_iters

    # --- LTM 特定参数 ---
    max_z_len = n_layers * 8  # 12 * 8 = 96
    z_dim = dim               # 1536

elif MODEL_SIZE == "LTM-1.3B":
    # --- 架构 ---
    dim = 2304
    n_layers = 18
    n_heads = 18
    n_kv_heads = 18 # LTM 默认 n_kv_heads = n_heads
    
    # --- 训练 (100B Tokens) ---
    max_tokens = 100_000_000_000
    warmup_tokens = int(max_tokens * 0.01) # 1% warmup
    
    max_iters = max_tokens // TOKENS_PER_ITERATION         # 100B / 2.097M ≈ 47684
    warmup_iters = warmup_tokens // TOKENS_PER_ITERATION   # 1B / 2.097M ≈ 477
    lr_decay_iters = max_iters
    
    # --- LTM 特定参数 ---
    max_z_len = n_layers * 8  # 18 * 8 = 144
    z_dim = dim               # 2304

else:
    raise ValueError(f"未知的 MODEL_SIZE: {MODEL_SIZE}。请选择 'LTM-421M' 或 'LTM-1.3B'。")

# -----------------------------------------------------------------------------
# Output and checkpointing settings
# -----------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = f"output/{MODEL_SIZE}_{timestamp}"
eval_interval = 1000  # Evaluate model every N iterations
log_interval = 1  # Log training progress every N iterations
eval_iters = 100  # Number of batches to use for evaluation
always_save_checkpoint = True  # If True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' (new model) or 'resume' (load from checkpoint)
ckpt_path = ''  # Path to checkpoint file when resuming training

# -----------------------------------------------------------------------------
# Model architecture
# -----------------------------------------------------------------------------
multiple_of = 32  # Hidden dimension is rounded to a multiple of this value
dropout = 0.0  # Dropout probability
# window_size = 256  # LTM 自己的滑动窗口参数
window_size = 2048

# -----------------------------------------------------------------------------
# Latent variable configuration
# -----------------------------------------------------------------------------
num_steps = 16  # Number of steps for posterior inference
inference_method = 'adamVI'  # Method used for posterior inference
initial_fast_lr = 0.3  # Initial learning rate for latent optimization
final_fast_lr = 0.34  # Final learning rate for latent optimization
fast_lr = 0.34  # Current learning rate for latent optimization
n_prior_layers = 0  # Number of layers in the prior network
n_cls_tokens = 0  # Number of classification tokens
# max_z_len = n_layers * 8  # Maximum length of latent sequence
# z_dim = dim  # Dimension of latent variables

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
# gradient_accumulation_steps = 8  # Number of steps to accumulate gradients (simulates larger batch)
learning_rate = 4e-4  # Maximum learning rate for model training
# max_iters = 60000  # Total number of training iterations (~30B tokens)
weight_decay = 1e-1  # L2 regularization
beta1 = 0.9  # AdamW beta1 parameter
beta2 = 0.95  # AdamW beta2 parameter
grad_clip = 1.0  # Gradient clipping threshold (disable if 0.0)

# Learning rate schedule settings
decay_lr = True  # Whether to use learning rate decay
# warmup_iters = 1000  # Number of warmup iterations
# lr_decay_iters = max_iters  # Number of iterations over which to decay learning rate
# min_lr = 4e-5  # Minimum learning rate (typically learning_rate/10)
min_lr = learning_rate / 10 # 4e-5

# -----------------------------------------------------------------------------
# System and hardware settings
# -----------------------------------------------------------------------------
device = "cuda"  # Device to use: 'cpu', 'cuda', 'cuda:0', etc.
dtype = "bfloat16"  # Data type: float32, bfloat16, or float16
compile = False  # Whether to use PyTorch 2.0 compilation for speed

# Create a dictionary of all configuration parameters
def get_config_dict():
    """Return a dictionary containing all configuration parameters."""
    globals_dict = globals()
    return {k: v for k, v in globals_dict.items() 
            if not k.startswith("_") and 
               not callable(v) and 
               k not in ('os', 'datetime', 'timestamp', 'TOKENS_PER_ITERATION')}