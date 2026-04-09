import csv
import os
from datetime import datetime


class MetricsWriter:
    """
    Writes one TSV row per call to .write(), appending to a single file
    for the entire training run. Safe to resume — if the file already
    exists, new rows are appended after existing ones.

    Columns: step, train_loss, train_bpc, val_loss, val_bpc, lr, elapsed_s, wall_time
    """

    FIELDS = ["step", "train_loss", "train_bpc", "val_loss", "val_bpc", "lr", "elapsed_s", "wall_time"]

    def __init__(self, path: str = "log.tsv"):
        self.path = path
        file_exists = os.path.isfile(path)

        self._f = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=self.FIELDS, delimiter="\t")

        # Write header only if the file is new
        if not file_exists:
            self._writer.writeheader()
            self._f.flush()

        print(f"Logging to '{self.path}' ({'appending' if file_exists else 'new file'})")

    def write(self, step: int, train_loss: float, bpc_train: float, 
              val_loss: float, bpc_val: float, lr: float, elapsed_s: float) -> None:

        self._writer.writerow({
            "step"       : step,
            "train_loss" : round(train_loss, 6),
            "train_bpc"  : round(bpc_train, 6),
            "val_loss"   : round(val_loss,   6),
            "val_bpc"    : round(bpc_val, 6),
            "lr"         : f"{lr:.2e}",
            "elapsed_s"  : round(elapsed_s, 2),
            "wall_time"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._f.flush()   # ensure row is written even if training crashes

    def close(self) -> None:
        self._f.close()
