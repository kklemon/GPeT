import argparse
import numpy as np

from pathlib import Path
from bpemb import BPEmb


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--dim', type=int, default=300)
parser.add_argument('--vocab_size', '--vs', type=int, default=10_000)

args = parser.parse_args()

tokenizer = BPEmb(lang=args.lang, vs=args.vocab_size, dim=args.dim)

input_path = Path(args.input)
output_path = Path(args.output)


ids = tokenizer.encode_ids(input_path.read_text())

np.save(output_path, np.array(ids))
