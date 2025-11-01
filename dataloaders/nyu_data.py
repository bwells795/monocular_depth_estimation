from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


print("building datasets")
ds = load_dataset("sayakpaul/nyu_depth_v2")

nyu_train_dataloader = DataLoader(ds["train"], batch_size=32)
nyu_test_dataloader = DataLoader(ds["validation"], batch_size=32)
