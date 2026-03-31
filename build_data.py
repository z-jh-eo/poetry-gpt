import torch


class BuildData:
    def __init__(self, text_path:str):
        self.text_path = text_path
        with open(self.text_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        print(f"Vocab size: {self.vocab_size}")
        print(f"Characters: {repr(''.join(self.chars))}")

        self.c2i = {ch: i for i, ch in enumerate(self.chars)}
        self.i2c = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s:str):
        return [self.c2i[c] for c in s]

    def decode(self, l:list):
        return ''.join([self.i2c[i] for i in l])

    def build(self, train_pc:float=0.9):
        data = torch.tensor(self.encode(self.text),
                            dtype=torch.long)
        n = int(train_pc * len(data))
        train_data = data[:n]
        val_data = data[n:]

        return train_data, val_data
