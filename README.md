# Recurrent Recurrent-VLN-BERT-Isaac

the base model is adapted from [[Paper & Appendices](https://arxiv.org/abs/2011.13922)] [[GitHub](https://github.com/YicongHong/Recurrent-VLN-BERT)]

Recurrent-VLN-BERT-Isaac is modified version of Recurrent-VLN-BERT, which has the following changes:
- Replace multiple viewpoint features with one feature vector per step.
- Action Prediction: Add a classification head on the state token instead of using attention weights for action selection.

The Original Recurrent-VLN-BERT and PREVALENT

- Original Model: Recurrent-VLN-BERT, built on OSCAR or PREVALENT, is designed for vision-and-language navigation (VLN). It processes instructions and visual inputs (multiple viewpoint features) recurrently, using the [CLS] token as the state, updated at each step with attention mechanisms to select actions.
- PREVALENT: A vision-and-language BERT variant with separate language and vision branches, using cross-attention for multimodal integration. For VLN, it processes instructions once and updates the state with visual inputs at each step.
- New Task: Unlike VLN, there’s no exploration; the model predicts expert actions from a single RGB-depth observation per step, making it an imitation learning problem.

Define the New Problem Requirements

- Inputs:
    - Instruction: A fixed natural language string (e.g., "Go to the kitchen").
    - RGB Image: Current observation as a PNG file.
    - Depth Image: Current depth map as a NumPy array.
    - Previous State: The [CLS] token’s output from the previous step.
    - Previous Action: The discrete action taken last step (optional, can be embedded if needed).
- Output: Probability distribution over three actions: "Move forward" (0), "Turn left" (1), "Turn right" (2).
- Recurrence: The state updates at each step, incorporating history implicitly via the transformer.

Pre-trained [PREVALENT](https://github.com/weituo12321/PREVALENT) weights
    - Download the `pytorch_model.bin` from [here](https://drive.google.com/drive/folders/1sW2xVaSaciZiQ7ViKzm_KbrLD_XvOq5y).

## Testing on the VLN-Go2-Matterport dataset

training command:
```
python src/train_navigation.py
```

testing command:
```
python src/test_navigation.py
```

## Evaluation on the simulator

testing on a single episode (e.g., episode 5):

```
python src/evaluate.py --task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02 --vlnbert_model_path=/home/prj21/fyp/Recurrent-VLN-BERT-Isaac/checkpoints/navigation_PREVALENT/2025-03-07_20-02-28/checkpoint_10000.pt --episode_index=5
```

testing on the n episodes (e.g., n=5):
```
python src/eval.py --model_path=/home/prj21/fyp/Recurrent-VLN-BERT-Isaac/checkpoints/navigation_PREVALENT/2025-03-07_20-02-28/checkpoint_10000.pt --num_episodes=5
```

```
The documentations are polished by ChatGPT.
Moreover, ChatGPT has been used in debugging and implementing some low-level algorithms such as those involving file operations.
```