import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from vlnbert.vlnbert_init import get_vlnbert_models
from param import args
from itertools import cycle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # renders plots to files without requiring a GUI
import matplotlib.pyplot as plt
import os
from torch.cuda.amp import GradScaler, autocast

from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

class NavigationAgent:
    def __init__(self, env, tokenizer, max_len=4000):
        self.env = env
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vln_bert = get_vlnbert_models().cuda()
        self.optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)

    def rollout(self, batch, train=True):
        torch.cuda.empty_cache()
        instr_encodings = batch["instr_encodings"].cuda()
        rgb = batch["rgb"].cuda()      # (batch, seq_len, 3, 224, 224)
        depth = batch["depth"].cuda()  # (batch, seq_len, 1, 224, 224)
        actions = batch["actions"].cuda()
        masks = batch["masks"].cuda()
        seq_len = rgb.size(1)

        lang_mask = (instr_encodings != self.tokenizer.pad_token_id).float().cuda()
        _, lang_output = self.vln_bert("language", instr_encodings, lang_mask=lang_mask)
        vis_mask = torch.ones(rgb.size(0), 1).cuda()

        losses = []
        predictions = []
        for t in range(seq_len):
            rgb_t = rgb[:, t, :, :, :]
            depth_t = depth[:, t, :, :, :]
            pooled_output, action_logits, next_lang_output = self.vln_bert("visual", lang_output, lang_mask=lang_mask, vis_mask=vis_mask, rgb=rgb_t, depth=depth_t)
            loss = F.cross_entropy(action_logits, actions[:, t], ignore_index=-1, reduction="none")
            losses.append(loss * masks[:, t].float())
            predictions.append(action_logits.argmax(dim=-1))
            lang_output = next_lang_output
        total_loss = torch.stack(losses).sum() / masks.sum()
        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
            self.optimizer.step()
        return total_loss.item(), torch.stack(predictions, dim=1), actions

    def train(self, n_iters, val_env, save_dir):
        writer = SummaryWriter(log_dir=f"runs/{args.name}")
        self.vln_bert.train()
        batch_iter = cycle(self.env)  # Infinite iterator over training batches
        num_warmup_steps = int(n_iters * args.warmup_fraction)
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=0
        )

        # Lists to store metrics for plotting
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')  # Initialize best validation loss
        
        for iter in tqdm(range(n_iters), desc="Training"):
            batch = next(batch_iter)
            loss, preds, gt = self.rollout(batch)
            train_losses.append(loss)
            
            # Compute training accuracy
            masks = batch["masks"].to(preds.device)  # Shape: (batch_size, sequence_length)
            valid_preds = preds[masks]               # 1D tensor of valid predictions
            valid_gt = gt[masks]                     # 1D tensor of valid ground truth
            if valid_preds.numel() > 0:              # Check if there are valid steps
                train_accuracy = (valid_preds == valid_gt).float().mean().item()
            else:
                train_accuracy = 0.0                 # No valid steps, set accuracy to 0
            
            # Log and display metrics
            tqdm.write(f"Iter {iter+1}/{n_iters}, Train Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            writer.add_scalar("Loss/train", loss, iter)
            writer.add_scalar("Accuracy/train", train_accuracy, iter)
            
            scheduler.step()
            
            # Periodic validation and checkpointing every 100 iterations
            if (iter + 1) % 100 == 0:
                val_loss, val_accuracy = self.validate(val_env)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                writer.add_scalar("Loss/val", val_loss, iter)
                writer.add_scalar("Accuracy/val", val_accuracy, iter)
                tqdm.write(f"Iter {iter+1}/{n_iters}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(save_dir, f"checkpoint_{iter+1}.pt")
                self.save(checkpoint_path)
                
                # Save best model if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_dir, "best_model.pt")
                    self.save(best_model_path)
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(save_dir, f"checkpoint_{n_iters}.pt")
        self.save(final_checkpoint_path)
        
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