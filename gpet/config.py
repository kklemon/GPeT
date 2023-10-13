from dataclasses import dataclass
from bpemb import BPEmb
    

@dataclass
class TokenizerConfig:
    lang: str
    dim: int
    vocab_size: int

    def load(self):
        return BPEmb(lang=self.lang, dim=self.dim, vs=self.vocab_size)


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 3_000


@dataclass
class ModelConfig:
    seq_len: int
    predict_embeds: bool = True
    intermediate_dim: int | None = None
    model_dim: int = 512
    num_layers: int = 8


@dataclass
class EvaluationConfig:
    num_samples_per_epoch: int = 10
    sample_prompt: str = "Deep Learning is "
    temperature: float = 1.0
