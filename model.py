import math
from dataclasses import dataclass

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.n_embd,
            config.n_heads,
            batch_first=True
        )
        self.ff   = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        )
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop(
            self.attn(
                query=self.ln1(x),
                key  =self.ln1(x),
                value=self.ln1(x),
                is_causal=True
            )[0]
        )

        x = x + self.drop(self.ff(self.ln2(x)))

        return(x)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int    = 12
    n_head: int     = 12
    n_embd: int     = 768
    dropout: float  = 0.1
    bias: bool      = True


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.voacb_size is not None
        assert config.block_size is not None
        self.config =  config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd)
            wpe = nn.Embedding(config.block_size, config.n_embd)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _  in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (slef.get_num_params()/1e6,))

    def get_num_params(self,  non_embedding=True):
        n_params = sum(p.numel()  for p in self.parameters())
        if  non_embedding:
            n_params -=  self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)




