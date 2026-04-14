import torch
import argparse

from build_data import BuildData
from model import GPT
from model import GPTConfig

CORPUS = "./extracted_sonnets.txt"
MOD = "./models/ckpt_step005000.pt"

# Load vocab
dataset = BuildData(CORPUS)

# Load config and rebuild model
ckpt = torch.load(MOD, map_location="cpu", weights_only=False)
config = ckpt["config"]
model = GPT(config)

state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(ckpt["model"])
model.eval()


def generate(prompt: str, max_new_tokens=1000, temperature=0.8, top_k=40):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blank-prompt", action="store_true")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=1_000)
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-k", "--top-k", type=int, default=20)
    args = parser.parse_args()

    if args.blank_prompt:
        print(
            generate(
                prompt="\n\n\n",
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )

    else:
        _prompt = "\n\n\n\
C'est la Mort qui console, hélas! et qui fait vivre;\n\
C'est le but de la vie, et c'est le seul espoir\n\
Qui, comme un élixir, nous monte et nous enivre,\n\
Et nous donne le coeur de marcher jusqu'au soir;\n"

        print(
            generate(
                prompt=_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )
