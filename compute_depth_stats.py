import os
import numpy as np
from tqdm import tqdm

def compute_depth_statistics(data_dir):
    min_depth = float('inf')  # Initialize to infinity
    max_depth = float('-inf')  # Initialize to negative infinity
    count = 0                 # Total number of depth values
    mean = 0.0                # Running mean
    M2 = 0.0                  # Sum of squared differences for variance

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
                # Update min and max
                min_depth = min(min_depth, depth.min())
                max_depth = max(max_depth, depth.max())
                # # Update running statistics for mean and standard deviation
                # for value in depth.flatten():
                #     count += 1
                #     delta = value - mean
                #     mean += delta / count
                #     delta2 = value - mean
                #     M2 += delta * delta2
                print(min_depth)
                print(max_depth)
                print("---")
    # Compute standard deviation
    if count < 2:
        std_depth = float('nan')  # Not enough data to compute std
    else:
        variance = M2 / (count - 1)
        std_depth = np.sqrt(variance)

    return min_depth, max_depth, mean, std_depth

# Run the computation
data_dir = "../VLN-Go2-Matterport/training_data"
min_depth, max_depth, mean_depth, std_depth = compute_depth_statistics(data_dir)
print(f"Min Depth: {min_depth:.2f} meters")
print(f"Max Depth: {max_depth:.2f} meters")
print(f"Mean Depth: {mean_depth:.2f} meters")
print(f"Std Depth: {std_depth:.2f} meters")