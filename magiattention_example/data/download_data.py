import os
import json
from datasets import load_dataset

dataset = load_dataset("openwebtext", num_proc=8, split='train')
dataset.to_json("./openwebtext.json", lines=True)
