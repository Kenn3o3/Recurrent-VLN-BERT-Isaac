import os
import numpy as np
from tqdm import tqdm

def compute_depth_statistics_sequential(data_dir):
    min_depth = float('inf')
    max_depth = float('-inf')
    count = 0
    mean = 0.0
    M2 = 0.0

    for episode in tqdm(os.listdir(data_dir), desc="Processing episodes"):
        episode_path = os.path.join(data_dir, episode)
        if not os.path.isdir(episode_path):
            continue
        depth_dir = os.path.join(episode_path, "depths")
        if not os.path.exists(depth_dir):
            continue
        for depth_file in os.listdir(depth_dir):
            if depth_file.endswith(".npy"):
                depth_path = os.path.join(depth_dir, depth_file)
                try:
                    depth = np.load(depth_path)
                    if depth.size == 0:
                        continue

                    min_depth = min(min_depth, depth.min())
                    max_depth = max(max_depth, depth.max())

                    n = depth.size
                    batch_mean = np.mean(depth)
                    batch_M2 = np.var(depth) * (n - 1)

                    if count == 0:
                        mean = batch_mean
                        M2 = batch_M2
                    else:
                        delta = batch_mean - mean
                        mean = mean + delta * n / (count + n)
                        M2 = M2 + batch_M2 + delta**2 * count * n / (count + n)
                    count += n
                except Exception as e:
                    print(f"Error processing {depth_path}: {e}")

    std_depth = np.nan if count < 2 else np.sqrt(M2 / (count - 1))
    return min_depth, max_depth, mean, std_depth

data_dir = "../VLN-Go2-Matterport/training_data"
min_depth, max_depth, mean_depth, std_depth = compute_depth_statistics_sequential(data_dir)
print(f"Min Depth: {min_depth:.2f} meters")
print(f"Max Depth: {max_depth:.2f} meters")
print(f"Mean Depth: {mean_depth:.2f} meters")
print(f"Std Depth: {std_depth:.2f} meters")