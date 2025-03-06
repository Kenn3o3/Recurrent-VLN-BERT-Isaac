import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

def extract_features(dataset_dir, output_dir):
    resnet = models.resnet50(pretrained=True).eval().cuda()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    os.makedirs(output_dir, exist_ok=True)
    for episode in os.listdir(dataset_dir):
        episode_path = os.path.join(dataset_dir, episode)
        rgb_dir = os.path.join(episode_path, "rgbs")
        depth_dir = os.path.join(episode_path, "depths")
        feat_dir = os.path.join(episode_path, "features")
        os.makedirs(feat_dir, exist_ok=True)
        for i, rgb_file in enumerate(sorted(os.listdir(rgb_dir))):
            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(depth_dir, f"depth_{i}.npy")
            rgb = Image.open(rgb_path).convert("RGB")
            rgb_tensor = transform(rgb).unsqueeze(0).cuda()
            depth = np.load(depth_path)[..., np.newaxis]  # HxWx1
            depth_tensor = transform(Image.fromarray(depth)).unsqueeze(0).cuda()
            with torch.no_grad():
                rgb_feat = resnet(rgb_tensor).squeeze().cpu().numpy()
                depth_feat = resnet(depth_tensor).squeeze().cpu().numpy()
            feat = np.concatenate([rgb_feat, depth_feat])
            np.save(os.path.join(feat_dir, f"feat_{i}.npy"), feat)

extract_features("../VLN-Go2-Matterport/training_data", "../VLN-Go2-Matterport/training_data_features")