import os
import numpy as np
from tqdm import tqdm

def compute_depth_statistics(data_dir):
    min_depth = float('inf')  # Initialize to infinity
    max_depth = float('-inf')  # Initialize to negative infinity
    all_depths = []  # List to collect all depth values for mean/std

    # Iterate over all episodes in the dataset
    for episode in tqdm(os.listdir(data_dir)):
        episode_path = os.path.join(data_dir, episode)
        if not os.path.isdir(episode_path):
            continue  # Skip non-directory files
        depth_dir = os.path.join(episode_path, "depths")
        if not os.path.exists(depth_dir):
            continue  # Skip episodes without depth directory
        # Process each depth file in the episode
        for depth_file in os.listdir(depth_dir):
            if depth_file.endswith(".npy"):
                depth_path = os.path.join(depth_dir, depth_file)
                depth = np.load(depth_path)  # Load depth map
                all_depths.append(depth)  # Store for mean/std calculation
                min_depth = min(min_depth, depth.min())  # Update global min
                max_depth = max(max_depth, depth.max())  # Update global max

    # Compute mean and standard deviation
    all_depths = np.concatenate(all_depths, axis=0)  # Combine all depth maps
    mean_depth = all_depths.mean()
    std_depth = all_depths.std()

    return min_depth, max_depth, mean_depth, std_depth

# Run the computation
data_dir = "../VLN-Go2-Matterport/training_data"
min_depth, max_depth, mean_depth, std_depth = compute_depth_statistics(data_dir)
print(f"Min Depth: {min_depth:.2f} meters")
print(f"Max Depth: {max_depth:.2f} meters")
print(f"Mean Depth: {mean_depth:.2f} meters")
print(f"Std Depth: {std_depth:.2f} meters")