import os
from env import NavigationBatch
from agent import NavigationAgent
from param import args
from vlnbert.vlnbert_init import get_tokenizer

data_dir = "../VLN-Go2-Matterport"
test_env = NavigationBatch(os.path.join(data_dir, "val"), batch_size=args.batchSize, splits=["val"], tokenizer=get_tokenizer(args))
agent = NavigationAgent(test_env, get_tokenizer(args))
agent.load(os.path.join("snap", args.name, "navigation_model.pt"))
loss, preds, gt = agent.test()
accuracy = (preds == gt).float().mean().item()
print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")