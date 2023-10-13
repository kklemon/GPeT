from pathlib import Path
from re import I
from typing import Any
from attr import dataclass
from einops import rearrange
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import numpy as np
import lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_lightning.loggers import WandbLogger
from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader

@dataclass
class TokenizerConfig:
    name: str
    dim: int
    vocab_size: int

    def load(self):
        return BPEmb(lang=self.name, dim=self.dim, vs=self.vocab_size)
    

class LanguageModelDataset(Dataset):
    def __init__(self, path, seq_len=256, overlap=0, debug=False):
        self.ids = np.load(path)
        self.seq_len = seq_len
        self.overlap = overlap
        self.debug = debug

        if debug:
            self.ids = self.ids[:100_000]

    def __len__(self):
        return (len(self.ids) - self.overlap - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len - self.overlap)
        end = start + self.seq_len
        
        x = self.ids[start:end]
        y = self.ids[start+1:end+1]

        return x, y
    

class LitLanguageModelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        tokenizer_config: TokenizerConfig,
        seq_len=256,
        overlap=0,
        batch_size: int = 128,
        debug: bool = False
    ):
        super().__init__()
        self.root = Path(root)
        self.tokenizer_config = tokenizer_config
        self.seq_len = seq_len
        self.overlap = overlap
        self.batch_size = batch_size
        self.debug = debug

        self.tokenizer = None

        self.train_data = None
        self.val_data = None
    
    def setup(self, stage: str) -> None:
        self.tokenizer = self.tokenizer_config.load()

        kwargs = dict(
            seq_len=self.seq_len,
            overlap=self.overlap,
            debug=self.debug
        )

        self.train_data = LanguageModelDataset(
            self.root / "train.npy",
            **kwargs
        )

        self.val_data = LanguageModelDataset(
            self.root / "valid.npy",
            **kwargs
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)


class LitGPeT(pl.LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        seq_len: int,
        predict_embeds: bool = True
    ):
        super().__init__()

        self.tokenizer_config = tokenizer_config
        self.tokenizer = self.tokenizer_config.load()
        self.seq_len = seq_len
        self.predict_embeds = predict_embeds

        self.register_buffer("embeds", torch.tensor(self.tokenizer.vectors))

        model_dim = 512
        num_layers = 8

        self.tok_embeds = nn.Embedding(self.tokenizer.vocab_size, model_dim)
        self.pos_embeds = nn.Embedding(self.seq_len, model_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )

        out_dim = self.tokenizer.dim if self.predict_embeds else self.tokenizer.vocab_size

        self.to_out = nn.Linear(model_dim, out_dim)

        self.register_buffer('attn_mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = self.tok_embeds(x)
        x = x + self.pos_embeds(torch.arange(x.shape[1], device=x.device))[None, :, :]

        x = self.encoder(x, is_causal=True, mask=self.attn_mask)

        x = self.to_out(x)

        logits = x @ self.embeds.T if self.predict_embeds else x

        return logits

    def step(self, batch, log_prefix):
        x, y = batch

        logits = self(x)

        loss = F.cross_entropy(
            rearrange(logits, 'b n d -> (b n) d'),
            rearrange(y, 'b n -> (b n)')
        )

        self.log(f'{log_prefix}/loss', loss, prog_bar=True)

        return loss

    def training_step(self, batch):
        return self.step(batch, 'train')
    
    def validation_step(self, batch):
        return self.step(batch, 'val')


tokenizer_config = TokenizerConfig(name="en", dim=50, vocab_size=50000)
seq_len = 256

data_module = LitLanguageModelDataModule(
    root="/data/datasets/nlp/wikitext-103/bpe.vs_50",
    tokenizer_config=tokenizer_config,
    seq_len=seq_len,
    overlap=0,
    batch_size=64,
    debug=False
)

model = LitGPeT(
    tokenizer_config=tokenizer_config,
    seq_len=seq_len,
    predict_embeds=False
)

trainer = pl.Trainer(
    devices=[1],
    logger=WandbLogger(project="gpet"),
)
trainer.fit(model, data_module)
