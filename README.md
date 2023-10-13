Generative Pretrained Embedding Transformer
===========================================

A small experiment on language modeling by predicting pretrained (sub-) word embeddings.

Usage
-----

After cloning, install the dependencies with Poetry and activate the Python environment:

```bash
poetry install
poetry shell
```

### Data Preparation

Use the `prepare_data.py` script to byte-pair encode text files:

```bash
python prepare_data.py train.txt train.npy
python prepare_data.py valid.txt valid.npy
```

By default, a pretrained BPE model with vocab size 10,000 and 300 dimensional embeddings will be used.

### Training

Train an embedding prediction language model:

```bash
python train.py fit \
    --config=config/base.yaml \
    --data.root=<data-dir> \
    --trainer.precision=16-mixed  # For mixed-precision training
```

Set `model.model_config.predict_embeds` to `False` to let the model predict logits over words directly and thus train a *coventional* LM:

```bash
python train.py fit \
    --config=config/base.yaml \
    --data.root=<data-dir> \
    --trainer.precision=16-mixed  # For mixed-precision training \
    --model.model_config.predict_embeds=false
```

Both variants will use the same loss formulation and sampling and can thus be directly compared.