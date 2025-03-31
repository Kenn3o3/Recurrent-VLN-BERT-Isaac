import torch
import os
import random
from env import NavigationBatch
from agent import NavigationAgent
from param import args
from vlnbert.vlnbert_init import get_tokenizer
import warnings
from datetime import datetime 

warnings.filterwarnings("ignore")
data_dir = "../VLN-Go2-Matterport"
# data_dir = "./"

all_episodes = os.listdir(os.path.join(data_dir, "training_data"))
random.seed(42)
random.shuffle(all_episodes)
train_size = int(0.8 * len(all_episodes))
val_size = int(0.2 * len(all_episodes))
print("val size: ", val_size)
print("train size: ", train_size)
train_episodes = all_episodes[:train_size]
val_episodes = all_episodes[train_size:train_size + val_size]
test_episodes = all_episodes[train_size + val_size:]
print("Ready to train!!!")
train_env = NavigationBatch(data_dir, batch_size=args.batchSize, episodes=train_episodes, tokenizer=get_tokenizer(args))
val_env = NavigationBatch(data_dir, batch_size=args.batchSize, episodes=val_episodes, tokenizer=get_tokenizer(args))

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("checkpoints", args.name, current_time)
os.makedirs(save_dir, exist_ok=True)

agent = NavigationAgent(train_env, get_tokenizer(args))
agent.train(args.iters, val_env, save_dir)