import torch
import os
import random
from env import NavigationBatch
from agent import NavigationAgent
from param import args
from vlnbert.vlnbert_init import get_tokenizer

feature_dir = "../VLN-Go2-Matterport"

all_episodes = os.listdir(os.path.join(feature_dir, "training_data"))
random.seed(42)
random.shuffle(all_episodes)
train_size = int(0.8 * len(all_episodes))
val_size = int(0.1 * len(all_episodes))
train_episodes = all_episodes[:train_size]
val_episodes = all_episodes[train_size:train_size + val_size]
test_episodes = all_episodes[train_size + val_size:]

train_env = NavigationBatch(feature_dir, batch_size=args.batchSize, episodes=train_episodes, tokenizer=get_tokenizer(args))
val_env = NavigationBatch(feature_dir, batch_size=args.batchSize, episodes=val_episodes, tokenizer=get_tokenizer(args))

agent = NavigationAgent(train_env, get_tokenizer(args))
agent.train(args.iters)

agent.save(os.path.join("checkpoints", args.name, "navigation_model.pt"))