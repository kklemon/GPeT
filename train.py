import torch

from gpet.cli import CLI
from gpet.lightning import LitLanguageModelDataModule, LitGPeT


torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    CLI(
        LitGPeT,
        LitLanguageModelDataModule,
        save_config_kwargs={"overwrite": True}
    )
