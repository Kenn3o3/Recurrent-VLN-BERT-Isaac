import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from vlnbert.vlnbert_init import get_vlnbert_models
from param import args
from itertools import cycle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class NavigationAgent:
    def __init__(self, env, tokenizer, max_len=4000):
        self.env = env
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vln_bert = get_vlnbert_models().cuda()
        self.optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)

    def rollout(self, batch, train=True):
        instr_encodings = batch["instr_encodings"].cuda()
        rgb = batch["rgb"].cuda()      # (batch, seq_len, 3, 224, 224)
        depth = batch["depth"].cuda()  # (batch, seq_len, 1, 224, 224)
        actions = batch["actions"].cuda()
        masks = batch["masks"].cuda()
        seq_len = rgb.size(1)

        state, lang_output = self.vln_bert("language", instr_encodings)
        lang_mask = (instr_encodings != self.tokenizer.pad_token_id).float().cuda()
        vis_mask = torch.ones(rgb.size(0), 1).cuda()

        losses = []
        predictions = []
        for t in range(seq_len):
            rgb_t = rgb[:, t, :, :, :]
            depth_t = depth[:, t, :, :, :]
            next_state, action_logits, _ = self.vln_bert("visual", lang_output, lang_mask=lang_mask, vis_mask=vis_mask, rgb=rgb_t, depth=depth_t)
            loss = F.cross_entropy(action_logits, actions[:, t], ignore_index=-1, reduction="none")
            losses.append(loss * masks[:, t].float())
            predictions.append(action_logits.argmax(dim=-1))
            lang_output = next_state

        total_loss = torch.stack(losses).sum() / masks.sum()
        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
            self.optimizer.step()
        return total_loss.item(), torch.stack(predictions, dim=1), actions

    def train(self, n_iters, val_env):
        writer = SummaryWriter(log_dir=f"runs/{args.name}")
        self.vln_bert.train()
        batch_iter = cycle(self.env)  # Infinite iterator over training batches
        
        # Lists to store metrics for plotting
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for iter in tqdm(range(n_iters), desc="Training"):
            batch = next(batch_iter)
            loss, preds, gt = self.rollout(batch)
            train_losses.append(loss)
            writer.add_scalar("Loss/train", loss, iter)
            
            # Periodic validation every 100 iterations
            if (iter + 1) % 100 == 0:
                val_loss, val_accuracy = self.validate(val_env)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                writer.add_scalar("Loss/val", val_loss, iter)
                writer.add_scalar("Accuracy/val", val_accuracy, iter)
                tqdm.write(f"Iter {iter+1}/{n_iters}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        writer.close()
        
        # Plotting at the end
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Iterations")
        plt.legend()
        plt.savefig(f"train_loss_{args.name}.png")
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(100, n_iters + 1, 100), val_losses, label="Validation Loss")
        plt.plot(range(100, n_iters + 1, 100), val_accuracies, label="Validation Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Metric Value")
        plt.title("Validation Metrics Over Iterations")
        plt.legend()
        plt.savefig(f"val_metrics_{args.name}.png")
        plt.close()

    def validate(self, val_env):
        self.vln_bert.eval()
        total_loss = 0
        total_correct = 0
        total_steps = 0
        with torch.no_grad():
            val_env.ix = 0  # Reset iterator
            try:
                while True:
                    batch = next(val_env)
                    loss, preds, gt = self.rollout(batch, train=False)
                    total_loss += loss * batch["masks"].sum().item()
                    total_correct += (preds == gt).sum().item()
                    total_steps += batch["masks"].sum().item()
            except StopIteration:
                pass
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        accuracy = total_correct / total_steps if total_steps > 0 else 0
        self.vln_bert.train()  # Restore training mode
        return avg_loss, accuracy

    def test(self):
        self.vln_bert.eval()
        total_loss = 0
        total_correct = 0
        total_steps = 0
        with torch.no_grad():
            self.env.ix = 0
            try:
                while True:
                    batch = next(self.env)
                    loss, preds, gt = self.rollout(batch, train=False)
                    total_loss += loss * batch["masks"].sum().item()
                    total_correct += (preds == gt).sum().item()
                    total_steps += batch["masks"].sum().item()
            except StopIteration:
                pass
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        accuracy = total_correct / total_steps if total_steps > 0 else 0
        return avg_loss, accuracy

    def save(self, path):
        torch.save(self.vln_bert.state_dict(), path)

    def load(self, path):
        self.vln_bert.load_state_dict(torch.load(path))