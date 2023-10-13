from pathlib import Path
from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from gpet.training import CosineWithWarmupLR
from gpet.config import EvaluationConfig, ModelConfig, TokenizerConfig, TrainingConfig
    

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
        batch_size: int = 64,
        num_workers: int = 4,
        debug: bool = False
    ):
        super().__init__()

        self.save_hyperparameters()

        self.root = Path(root)
        self.tokenizer_config = tokenizer_config
        self.seq_len = seq_len
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
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

    def create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_data, shuffle=True)
    
    def val_dataloader(self):
        return self.create_dataloader(self.val_data)


class LitGPeT(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer_config: TokenizerConfig,
        evaluation_config: EvaluationConfig,
    ):
        super().__init__()

        assert not model_config.intermediate_dim or model_config.predict_embeds

        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer_config = tokenizer_config
        self.evaluation_config = evaluation_config

        self.tokenizer = self.tokenizer_config.load()

        self.register_buffer("embeds", torch.tensor(self.tokenizer.vectors))
        self.register_buffer(
            'attn_mask', nn.Transformer.generate_square_subsequent_mask(self.model_config.seq_len)
        )
        self.tok_embeds = nn.Embedding(self.tokenizer.vocab_size, self.model_config.model_dim)
        self.pos_embeds = nn.Embedding(self.model_config.seq_len, self.model_config.model_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.model_config.model_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=self.model_config.num_layers
        )

        if self.model_config.predict_embeds:
            if self.model_config.intermediate_dim:
                self.proj_word_vecs = nn.Linear(self.tokenizer.dim , self.model_config.intermediate_dim)
                self.to_out = nn.Linear(self.model_config.model_dim, self.model_config.intermediate_dim)
            else:
                self.proj_word_vecs = nn.Identity()
                self.to_out = nn.Linear(self.model_config.model_dim, self.tokenizer.dim)
        else:
            self.to_out = nn.Linear(self.model_config.model_dim, self.tokenizer.vocab_size, bias=False)

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
            eps=1e-6,
            betas=(0.9, 0.999),
        )

        # If the trainer is already available at this point, the number of training
        # steps may be derived from max_epochs
        scheduler = CosineWithWarmupLR(
            opt,
            training_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=self.training_config.warmup_steps,
        )
        return ([opt], [{"scheduler": scheduler, "interval": "step"}])

    def forward(self, x):
        x = self.tok_embeds(x)
        x = x + self.pos_embeds(torch.arange(x.shape[1], device=x.device))[None, :, :]

        x = self.encoder(x, is_causal=True, mask=self.attn_mask)

        x = self.to_out(x)

        if self.model_config.predict_embeds:
            word_vecs = self.proj_word_vecs(self.embeds)
            logits = x @ word_vecs.T
        else:
            logits = x

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
    
    def on_train_epoch_end(self) -> None:
        if not self.evaluation_config.num_samples_per_epoch or self.global_rank != 0:
            return
        
        print(flush=True)
        
        for sample in self.sample(
            self.evaluation_config.sample_prompt,
            self.evaluation_config.num_samples_per_epoch,
            self.model_config.seq_len,
            temperature=self.evaluation_config.temperature
        ):
            print(sample)

    @torch.no_grad()
    def sample(self, prompt, num_samples, seq_len, temperature=1.0):
        assert seq_len <= self.model_config.seq_len

        ids = torch.tensor(self.tokenizer.encode_ids(prompt), device=self.device).unsqueeze(0)
        ids = ids.repeat(num_samples, 1)

        while ids.shape[-1] < seq_len:
            logits = self(ids) / temperature
            probas = torch.softmax(logits[:, -1], dim=-1)

            next_token = torch.multinomial(probas, num_samples=1)

            ids = torch.cat([ids, next_token], dim=-1)

        return [
            ' '.join(self.tokenizer.decode_ids(ids[i].tolist()))
            for i in range(num_samples)
        ]
