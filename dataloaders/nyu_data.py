from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


# class NYUData(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.ds = load_dataset("sayakpaul/nyu_depth_v2")
#
#     def __getitem__(self, index) -> _T_co:
#         return super().__getitem__(index)
#
#     def process_color(self, depth, d_min=None, d_max=None):
#         """
#         convert depth into a depth relative
#         """
#         cmap = plt.cm.viridis
#         if d_min is None:
#             d_min = np.min(depth)
#         if d_max is None:
#             d_max = np.max(depth)
#         depth_relative = (depth - d_min) / (d_max - d_min)
#         return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
#
#     def merge_into_row(self, input, depth_target):
#         input = np.array(input)
#         depth_target = np.squeeze(np.array(depth_target))
#
#         d_min = np.min(depth_target)
#         d_max = np.max(depth_target)
#         depth_target_col = self.process_color(depth_target, d_min, d_max)
#         img_merge = np.hstack([input, depth_target_col])
#
#         return img_merge
#
#     def visualize_random_sample(self):
#         """Create a visualization of a random set of images"""
#         random_indices = np.random.sample(self.ds.shape)
#         plt.figure(figsize=(15, 6))
#
#         for i, idx in enumerate(random_indices):
#             ax = plt.subplot(3, 3, i + 1)
#             image_viz = merge_into_row(
#                 train_set[idx]["image"], train_set[idx]["depth_map"]
#             )
#             plt.imshow(image_viz.astype("uint8"))
#             plt.axis("off")
#
#         plt.show()
print("building datasets")
ds = load_dataset("sayakpaul/nyu_depth_v2")

nyu_train_dataloader = DataLoader(ds["train"], batch_size=32)
nyu_test_dataloader = DataLoader(ds["validation"], batch_size=32)
