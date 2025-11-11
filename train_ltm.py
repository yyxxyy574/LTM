"""
Main training script for Latent Thought Language Model
"""
import math
import os
import time
from contextlib import nullcontext
from functools import partial
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Import configuration
import config
from model import LatentThoughtModel, LTMConfig
from optimizer import PosteriorOptimizer
from owt import Task

def main():
    """Main training function."""
    
    # -----------------------------------------------------------------------------
    # Distributed Training Setup
    # -----------------------------------------------------------------------------
    
    # Check if this is a distributed data parallel (DDP) run
    ddp = int(os.environ.get("RANK", -1)) != -1
    print(f"Using DDP for training: {ddp}")
    
    # Local variables to track the current training state
    iter_num = 0
    best_val_loss = 1e9
    ddp_world_size = 1
    gradient_accumulation_steps = config.gradient_accumulation_steps
    device = config.device
    
    if ddp:
        # Initialize the distributed process group
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])  # Global rank of this process
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Local rank on this node
        ddp_world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes
        device = f"cuda:{ddp_local_rank}"
        print(f"DDP setup complete. Using device: {device}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 1  # Process responsible for logging and checkpoints
        seed_offset = ddp_rank  # Each process gets a different seed
        
        # Scale down gradient accumulation steps proportionally to world size
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
        print(f"Adjusted gradient accumulation steps: {gradient_accumulation_steps}")
    else:
        print("Single GPU training (no DDP)")
        master_process = True
        seed_offset = 0
    
    # Calculate tokens processed per iteration for logging
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config.batch_size * config.max_seq_len
    if master_process:
        print(f"Tokens per iteration: {tokens_per_iter:,}")
        print(f"  = {gradient_accumulation_steps} accumulation steps * {ddp_world_size} processes * {config.batch_size} batch size * {config.max_seq_len} sequence length")
    
    # Create output directories on the master process
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)    
    # -----------------------------------------------------------------------------
    # Initialization and Setup
    # -----------------------------------------------------------------------------
    
    # Set random seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Device and precision setup
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32, 
        "bfloat16": torch.bfloat16, 
        "float16": torch.float16
    }[config.dtype]
    
    # Context manager for mixed precision training
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    
    # -----------------------------------------------------------------------------
    # Data Loading Setup
    # -----------------------------------------------------------------------------
    
    # Set up data iterator with latent variables
    iter_batches = partial(
        # Task.iter_batches_with_latents,
        Task.iter_batches_packed_with_latents,
        data_dir=config.DATA_CACHE_DIR,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        device=device,
        num_workers=0,
    )
    
    # -----------------------------------------------------------------------------
    # Model Initialization
    # -----------------------------------------------------------------------------
    
    # Define model architecture parameters
    model_args = dict(
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        vocab_size=config.vocab_size,
        multiple_of=config.multiple_of,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        n_prior_layers=config.n_prior_layers,
        n_cls_tokens=config.n_cls_tokens,
        window_size=config.window_size,
        use_liger=True,  # Enable LIGER (Learned Implicit Generator) mode
        max_z_len=config.max_z_len,
        use_z_pos_emb=True,  # Use positional embeddings for latent variables
    )
    
    if config.init_from == "scratch":
        # Initialize a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = LTMConfig(**model_args)
        model = LatentThoughtModel(gptconf)
        print(model)
    elif config.init_from == "resume":
        print(f"Resuming training from checkpoint: {config.ckpt_path}")
        # Resume training from a checkpoint
        checkpoint = torch.load(config.ckpt_path, map_location=device)
        
        # Use architecture parameters from checkpoint
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        
        # Create model with checkpoint configuration
        gptconf = LTMConfig(**model_args)
        model = LatentThoughtModel(gptconf)
        
        # Load model weights from checkpoint, handling DDP prefixes if present
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        
        # Restore training state
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    
    # Move model to appropriate device
    model.to(device)
    
    # -----------------------------------------------------------------------------
    # Optimizer and Precision Setup
    # -----------------------------------------------------------------------------
    
    # Set up gradient scaler for mixed precision training (no-op if not float16)
    scaler = torch.amp.GradScaler(enabled=(config.dtype == "float16"), device='cuda')
    
    # Initialize optimizer with weight decay
    optimizer = model.configure_optimizers(
        config.weight_decay, 
        config.learning_rate, 
        (config.beta1, config.beta2), 
        device_type
    )
    
    # Load optimizer state if resuming from checkpoint
    if config.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # Free up memory
    
    # Compile model for performance if enabled (requires PyTorch 2.0+)
    if config.compile:
        print("Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    
    # -----------------------------------------------------------------------------
    # Distributed Training Wrap-up
    # -----------------------------------------------------------------------------
    
    # Wrap model in DDP container for distributed training
    if ddp:
        # Ignore the `freqs_cis` buffer for DDP broadcasting (NCCL doesn't support ComplexFloat)
        prefix = "_orig_mod." if config.compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # -----------------------------------------------------------------------------
    # Model and Posterior Optimizer Setup
    # -----------------------------------------------------------------------------
    
    print(f"Training configuration: steps={config.num_steps}, layers={config.n_layers}, "
          f"z_len={config.max_z_len}, dim={config.dim}, heads={config.n_heads}")
    
    # Get raw model by unwrapping DDP container if needed
    raw_model = model.module if ddp else model
    
    # Initialize posterior optimizers for training and evaluation
    posterior_optimizer = PosteriorOptimizer(
        model=raw_model, 
        inference_method=config.inference_method, 
        num_steps=config.num_steps, 
        max_z_len=config.max_z_len, 
        z_dim=config.z_dim, 
        lr=config.fast_lr, 
        eval_mode=False
    )
    
    posterior_optimizer_test = PosteriorOptimizer(
        model=raw_model, 
        inference_method=config.inference_method, 
        num_steps=config.num_steps, 
        max_z_len=config.max_z_len, 
        z_dim=config.z_dim, 
        lr=config.fast_lr, 
        eval_mode=True
    )
    
    # -----------------------------------------------------------------------------
    # Training Utilities
    # -----------------------------------------------------------------------------
    
    def estimate_loss(lr=None):
        """
        Estimate loss on validation set.
        """
        loss_out = {}
        ppl_out = {}
        kl_out = {}
    
        model.eval()  # Set model to evaluation mode
        for split in ["val"]:
            batch_iter = iter_batches(split=split, batch_size=16)
            losses = torch.zeros(config.eval_iters)  # Track losses over evaluation iterations
            ppl_list = torch.zeros(config.eval_iters)  # Track perplexities
            kl_list = torch.zeros(config.eval_iters)  # Track KL divergences
            
            for k in range(config.eval_iters):
                # Get next batch
                X, Y, Z = next(batch_iter)
                
                # Optimize latent variables for this batch
                Z, ppl, kl_avg, nlkhd = posterior_optimizer_test.step(
                    data=[X, Y, Z], 
                    ctx=ctx, 
                    scaler=scaler, 
                    steps=config.num_steps, 
                    lr=lr
                )
                
                # Forward pass with optimized latents
                with ctx:
                    logits = model(X, Z, Y)
                    loss = raw_model.last_loss
                
                # Record metrics
                losses[k] = loss.item()
                ppl_list[k] = ppl.item()
                kl_list[k] = kl_avg.item()
                
                # Clean up to avoid OOM issues
                del X, Y, Z, logits, loss, ppl, kl_avg, nlkhd
                torch.cuda.empty_cache() 
                
            # Compute average metrics
            loss_out[split] = losses.mean()
            ppl_out[split] = ppl_list.mean()
            kl_out[split] = kl_list.mean()
        
        model.train()  # Set model back to training mode
        torch.cuda.empty_cache()
        return loss_out, ppl_out, kl_out
    
    def get_lr(it):
        """
        Get learning rate for current iteration based on warmup and cosine decay.
        """
        # 1) Linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    def fast_lr_linear_decay(epoch):
        """
        Linearly interpolate fast learning rate between initial and final values.
        """
        return config.initial_fast_lr + epoch / (config.lr_decay_iters - 1) * (config.final_fast_lr - config.initial_fast_lr)
    
    # -----------------------------------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------------------------------
    
    # Initialize training
    train_batch_iter = iter_batches(split="train")
    X, Y, Z = next(train_batch_iter)
    t0 = time.time()
    local_iter_num = 0  # Iterations in current process
    running_mfu = -1.0  # Model flops utilization (efficiency metric)
    print(f"Starting training loop, max iterations: {config.max_iters}")
    
    while True:
        # Determine learning rates for this iteration
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate  # Model learning rate
        current_lr = fast_lr_linear_decay(iter_num)  # Latent variable learning rate
    
        # Update optimizer learning rates
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
        # -----------------------------------------------------------------------------
        # Evaluation and Checkpointing
        # -----------------------------------------------------------------------------
        
        # Evaluate model and save checkpoint periodically
        if iter_num % config.eval_interval == 0 and master_process:
            losses, ppl_out, kl_out = estimate_loss(current_lr)
            print(f"Step {iter_num}: val loss {losses['val']:.4f}, val PPL {ppl_out['val']:.4f}, val KL {kl_out['val']:.4f}")
    
            # Save checkpoint if validation loss improved or if always_save_checkpoint is True
            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config.get_config_dict(),
                        'rng_state': torch.random.get_rng_state()
                    }
                    ckpt_path = os.path.join(config.out_dir, f"ckpt_{iter_num}.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)
    
        # -----------------------------------------------------------------------------
        # Forward and Backward Pass
        # -----------------------------------------------------------------------------
        
        # Forward and backward passes with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            # For distributed training, only synchronize gradients on the last micro-step
            if ddp:
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
            
            # Optimize latent variables for current batch
            Z, ppl, kl, _ = posterior_optimizer.step(
                data=[X, Y, Z], 
                ctx=ctx, 
                scaler=scaler, 
                steps=config.num_steps, 
                lr=current_lr
            )
            
            # Forward pass with optimized latents
            with ctx:
                logits = model(X, Z, Y)
                loss = raw_model.last_loss
                loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                    
            # Prefetch next batch asynchronously while GPU is busy
            X, Y, Z = next(train_batch_iter)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
        
        # -----------------------------------------------------------------------------
        # Gradient Processing and Optimizer Step
        # -----------------------------------------------------------------------------
        
        # Apply gradient clipping if enabled
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)  # Unscale gradients for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        # Update model parameters
        scaler.step(optimizer)
        scaler.update()
        
        # Clear gradients to free memory
        optimizer.zero_grad(set_to_none=True)
    
        # -----------------------------------------------------------------------------
        # Logging and Timing
        # -----------------------------------------------------------------------------
        
        # Calculate timing and log progress
        t1 = time.time()
        dt = t1 - t0  # Time for this iteration
        t0 = t1
        
        if iter_num % config.log_interval == 0 and master_process:
            # Get loss as float, scale up due to gradient accumulation
            lossf = loss.item() * gradient_accumulation_steps
            
            # Calculate model flops utilization (efficiency metric)
            if local_iter_num >= 5:  # Skip first few iterations for warm-up
                mfu = raw_model.estimate_mfu(config.batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                
            # Log training progress
            print(
                f"{iter_num} | loss {lossf:.4f} | ppl {ppl:.4f} | kl {kl:.4f} | "
                f"lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        
        # Update iteration counters
        iter_num += 1
        local_iter_num += 1
    
        # Check for termination condition
        if iter_num > config.max_iters:
            print(f"Training completed after {iter_num} iterations.")
            break
    
    # -----------------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------------
    
    # Clean up distributed training resources
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()