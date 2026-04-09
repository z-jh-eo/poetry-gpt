import torch
from build_data import BuildData
from model import GPT

SAVE_DIR = "/content/drive/MyDrive/gpt-poetry"

# Load vocab
dataset = BuildData("./full-text.txt")

# Load config and rebuild model
ckpt = torch.load(f"{SAVE_DIR}/checkpoints/ckpt_step030000.pt",
                  map_location="cpu")
config = ckpt["config"]
model  = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()

# Generate
def generate(prompt: str, max_new_tokens=300, temperature=0.8, top_k=40):
    encoded = torch.tensor(
        dataset.encode(prompt), dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return dataset.decode(out[0].tolist())

print(generate("\n"))                          # blank prompt
