import os
from env import NavigationBatch
from agent import NavigationAgent
from param import args
from vlnbert.vlnbert_init import get_tokenizer
import glob  # Added import for glob

import warnings
warnings.filterwarnings("ignore")

def find_latest_model(checkpoints_dir, experiment_name):
    """
    Find the latest best model or checkpoint from the most recent datetime subfolder.
    
    Args:
        checkpoints_dir (str): Base directory for checkpoints (e.g., "checkpoints").
        experiment_name (str): Experiment name (e.g., "navigation_PREVALENT").
    
    Returns:
        str: Path to the latest model file.
    
    Raises:
        FileNotFoundError: If no datetime directories or checkpoint files are found.
    """
    experiment_path = os.path.join(checkpoints_dir, experiment_name)
    # Get all datetime subfolders and sort lexicographically (latest last)
    datetime_dirs = sorted(glob.glob(os.path.join(experiment_path, "*/")))
    if not datetime_dirs:
        raise FileNotFoundError(f"No datetime directories found in {experiment_path}.")
    
    latest_datetime_dir = datetime_dirs[-1]  # Most recent subfolder
    best_model_path = os.path.join(latest_datetime_dir, "best_model.pt")
    
    if os.path.exists(best_model_path):
        return best_model_path
    else:
        # Get all checkpoints and sort by iteration number
        checkpoint_files = sorted(
            glob.glob(os.path.join(latest_datetime_dir, "checkpoint_*.pt")),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {latest_datetime_dir}.")
        return checkpoint_files[-1]  # Latest checkpoint

data_dir = "./"
all_episodes = os.listdir(os.path.join(data_dir, "training_data"))
print(f"Testing on {len(all_episodes)} episodes")
test_env = NavigationBatch(data_dir, batch_size=args.batchSize, episodes=all_episodes, tokenizer=get_tokenizer(args))
agent = NavigationAgent(test_env, get_tokenizer(args))

# Load the latest model
model_path = find_latest_model("checkpoints", args.name)
print("model path: ", model_path)
agent.load(model_path)

loss, accuracy = agent.test()
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")