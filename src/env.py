import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from utils import Tokenizer, pad_instr_tokens

class NavigationBatch:
    def __init__(self, feature_dir, batch_size=64, episodes=None, tokenizer=None):
        """
        Initialize the data loader with a list of episode directories.
        
        Args:
            feature_dir (str): Base directory (e.g., "VLN-Go2-Matterport").
            batch_size (int): Number of episodes per batch.
            episodes (list): List of episode directory names (e.g., ["2025-03-04_22-48-51_scene_QUCTc6BB5sX_episode_1053", ...]).
            tokenizer (Tokenizer): Tokenizer for instruction encoding.
        """
        self.feature_dir = feature_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data = []
        
        # Default to all episodes if none provided
        if episodes is None:
            episodes = os.listdir(os.path.join(feature_dir, "training_data"))
        
        # Load data for each episode
        for episode in episodes:
            episode_path = os.path.join(feature_dir, "training_data", episode)
            with open(os.path.join(episode_path, "instructions.txt")) as f:
                instr = f.read().strip()
            with open(os.path.join(episode_path, "actions.txt")) as f:
                actions = [line.strip() for line in f.readlines()]
            instr_tokens = tokenizer.tokenize(instr)
            padded_tokens, _ = pad_instr_tokens(instr_tokens, args.maxInput)
            instr_encoding = tokenizer.convert_tokens_to_ids(padded_tokens)
            rgb_files = sorted([f for f in os.listdir(os.path.join(episode_path, "rgbs")) if f.endswith(".png")])
            depth_files = sorted([f for f in os.listdir(os.path.join(episode_path, "depths")) if f.endswith(".npy")])
            rgb_images = [self.rgb_transform(Image.open(os.path.join(episode_path, "rgbs", f))) for f in rgb_files]
            depth_maps = [self.depth_transform(np.load(os.path.join(episode_path, "depths", f))) for f in depth_files]
            self.data.append({
                "instr_id": episode,
                "instr_encoding": instr_encoding,
                "rgb": rgb_images,      # List of (3, 224, 224) tensors
                "depth": depth_maps,    # List of (1, 224, 224) tensors
                "actions": [self.action_to_idx(a) for a in actions]
            })
        self.ix = 0

    def action_to_idx(self, action):
        """Map action strings to indices."""
        return {"Move forward": 0, "Turn left": 1, "Turn right": 2}[action]

    def __iter__(self):
        return self

    def __next__(self):
        """Yield the next batch of data."""
        if self.ix >= len(self.data):
            self.ix = 0
            raise StopIteration
        batch = self.data[self.ix:self.ix + self.batch_size]
        self.ix += self.batch_size
        return self._collate(batch)

    def _collate(self, batch):
        """Collate batch data, padding sequences to the maximum length in the batch."""
        max_len = max(len(item["rgb"]) for item in batch)
        instr_encodings = torch.stack([torch.tensor(item["instr_encoding"], dtype=torch.long) for item in batch])
        rgb = [torch.stack(item["rgb"] + [torch.zeros(3, 224, 224)] * (max_len - len(item["rgb"]))) for item in batch]
        depth = [torch.stack(item["depth"] + [torch.zeros(1, 224, 224)] * (max_len - len(item["depth"]))) for item in batch]
        actions = [torch.tensor(item["actions"] + [-1] * (max_len - len(item["actions"])), dtype=torch.long) for item in batch]
        masks = [torch.ones(len(item["actions"]), dtype=torch.bool) + torch.zeros(max_len - len(item["actions"]), dtype=torch.bool) for item in batch]
        return {
            "instr_encodings": instr_encodings,         # (batch, maxInput)
            "rgb": torch.stack(rgb),                    # (batch, max_len, 3, 224, 224)
            "depth": torch.stack(depth),                # (batch, max_len, 1, 224, 224)
            "actions": torch.stack(actions),            # (batch, max_len)
            "masks": torch.stack(masks)                 # (batch, max_len)
        }

    @staticmethod
    def rgb_transform(img):
        """Transform for RGB images."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)

    @staticmethod
    def depth_transform(depth):
        """Transform for depth maps (assumes 2D NumPy array)."""
        depth = torch.from_numpy(depth).float() / 10.0  # Normalize by assumed max_depth=10.0
        depth = depth.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
        depth = F.interpolate(depth, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        return depth                                    # (1, 224, 224)