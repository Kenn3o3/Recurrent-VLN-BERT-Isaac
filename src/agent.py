import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from vlnbert.vlnbert_init import get_vlnbert_models
from param import args

class NavigationAgent:
    def __init__(self, env, tokenizer, max_len=4000):
        self.env = env
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vln_bert = get_vlnbert_models().cuda()
        self.optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)

    def rollout(self, train=True):
        batch = next(self.env)
        instr_encodings = batch["instr_encodings"].cuda()
        rgb = batch["rgb"].cuda()      # (batch, seq_len, 3, 224, 224)
        depth = batch["depth"].cuda()  # (batch, seq_len, 1, 224, 224)
        actions = batch["actions"].cuda()
        masks = batch["masks"].cuda()
        seq_len = rgb.size(1)

        # Initialize state with language
        state, lang_output = self.vln_bert("language", instr_encodings)
        lang_mask = (instr_encodings != self.tokenizer.pad_token_id).float().cuda()
        vis_mask = torch.ones(rgb.size(0), 1).cuda()

        losses = []
        predictions = []
        for t in range(seq_len):
            rgb_t = rgb[:, t, :, :, :]    # (batch, 3, 224, 224)
            depth_t = depth[:, t, :, :, :]  # (batch, 1, 224, 224)
            next_state, action_logits, _ = self.vln_bert("visual", lang_output, lang_mask=lang_mask, vis_mask=vis_mask, rgb=rgb_t, depth=depth_t)
            loss = F.cross_entropy(action_logits, actions[:, t], ignore_index=-1, reduction="none")
            losses.append(loss * masks[:, t].float())
            predictions.append(action_logits.argmax(dim=-1))
            lang_output = next_state  # Update state

        total_loss = torch.stack(losses).sum() / masks.sum()
        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
            self.optimizer.step()
        return total_loss.item(), torch.stack(predictions, dim=1), actions

    def train(self, n_iters):
        self.vln_bert.train()
        for iter in range(n_iters):
            loss, _, _ = self.rollout()
            print(f"Iter {iter+1}/{n_iters}, Loss: {loss:.4f}")

    def test(self):
        self.vln_bert.eval()
        with torch.no_grad():
            loss, preds, gt = self.rollout(train=False)
        return loss, preds, gt

    def save(self, path):
        torch.save(self.vln_bert.state_dict(), path)

    def load(self, path):
        self.vln_bert.load_state_dict(torch.load(path))