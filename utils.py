from attr import dataclass
from bpemb import BPEmb
    

@dataclass
class TokenizerConfig:
    name: str
    dim: int
    vocab_size: int

    def load(self):
        return BPEmb(lang=self.name, dim=self.dim, vs=self.vocab_size)
