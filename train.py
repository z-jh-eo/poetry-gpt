import os
import math
import time

import torch
from build_data import BuildData
from model import GPTConfig, GPT

torch.manual_seed(42)


dataset = BuildData("./full-text.txt")
train_data, val_data = dataset.build()


device_type = "cuda" if torch.cuda.is_available()\
                     else "cpu"
device = torch.device(device_type)
print(f"Using device: {device}")


# --- Hyperparameters ---
block_size   = 256
batch_size   = 64
max_steps    = 30_000
warmup_steps = 500
max_lr       = 3e-4
grad_clip    = 1.0
causal_mask  = "flash"

eval_interval       = 200
sample_interval     = 2_000
checkpoint_interval = 5_000
eval_batches        = 20

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples",     exist_ok=True)



config = GPTConfig(
    vocab_size  = dataset.vocab_size,
    block_size  = block_size,
    n_layer     = 6,
    embd_d      = 256,
    dropout     = 0.1,
    bias        = True,
    causal_mask = causal_mask,
)

model = GPT(config).to(device)

use_compile = (device_type == "cuda")
if use_compile:
    print("Compiling model...")
    model = torch.compile(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)


def get_lr(step:int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr


def get_batch(split:str):
    data = train_data if split == "train" else val_data
    # Sample random starting positions
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i        : i + block_size    ] for i in ix])
    y = torch.stack([data[i + 1    : i + block_size + 1] for i in ix])
    if device == "cuda":
        # pin arrays x,y, which allows us to move them 
        # to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), \
                y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_val_loss() -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch("val")
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) #average loss over eval batches


def generate_sample(prompt:str = "\n", max_new_tokens:int = 300,
                    temperature:float = 0.8, top_k:int = 40) -> str:
    model.eval()
    encoded = torch.tensor(
        dataset.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)
    
    with torch.no_grad():
        out = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    model.train()
    return dataset.decode(out[0].tolist())


_train_loss_window: list[float] = []
 
def log(step: int, val_loss: float, t_elapsed: float) -> None:
    train_loss = sum(_train_loss_window) / max(len(_train_loss_window), 1)
    bpc_train  = train_loss / math.log(2)
    bpc_val    = val_loss   / math.log(2)
    lr_now     = optimizer.param_groups[0]["lr"]
    print(
        f"step {step:6d} | "
        f"train loss {train_loss:.4f} (bpc {bpc_train:.3f}) | "
        f"val loss {val_loss:.4f} (bpc {bpc_val:.3f}) | "
        f"lr {lr_now:.2e} | "
        f"{t_elapsed:.1f}s"
    )
    _train_loss_window.clear()




print(f"\nStarting training: {max_steps} steps, causal_mask='{causal_mask}'\n")
model.train()
t_start = time.time()
 
for step in range(max_steps):
 
    # 1. Learning rate update
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
 
    # 2. Forward + loss
    x, y = get_batch("train")
    logits, loss = model(x, targets=y)
 
    # 3. Backward
    optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()
    loss.backward()
 
    # 4. Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
 
    # 5. Parameter update
    optimizer.step()
 
    _train_loss_window.append(loss.item())
 
    # ── Logging ──────────────────────────────────────────────────────────────
    if step % eval_interval == 0:
        val_loss = estimate_val_loss()
        log(step, val_loss, time.time() - t_start)
        t_start = time.time()
 
    # ── Sample generation ────────────────────────────────────────────────────
    if step % sample_interval == 0:
        sample = generate_sample()
        print("─" * 60)
        print(sample)
        print("─" * 60)
        with open(f"samples/sample_step{step:06d}.txt", "w") as f:
            f.write(sample)
 
    # ── Checkpointing ────────────────────────────────────────────────────────
    if step % checkpoint_interval == 0 and step > 0:
        ckpt_path = f"checkpoints/ckpt_step{step:06d}.pt"
        torch.save({
            "step"      : step,
            "model"     : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "config"    : config,
            "val_loss"  : val_loss if step % eval_interval == 0 else None,
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")
 
print("\nTraining complete.")
