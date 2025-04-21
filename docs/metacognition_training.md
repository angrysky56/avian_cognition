# Metacognition Module Training

This document describes how to train the Metacognition module for the Avian Cognition architecture. The Metacognition module enables the model to estimate confidence in its predictions by determining when it might be wrong.

## Overview

The Metacognition module is trained to predict whether the main model's prediction for a given hidden state is correct. It operates as a confidence estimator, taking the model's hidden state as input and outputting a scalar value between 0 and 1 representing confidence.

## Step 1: Generate Training Data

First, we need to generate training data by running a pretrained language model and extracting:
1. Hidden states (from the model's last layer)
2. Correctness labels (whether the model's prediction was correct)

Use the `generate_metacognition_data.py` script:

```bash
# Navigate to project root
cd /path/to/avian_cognition

# Generate data using pretrained model
python scripts/generate_metacognition_data.py \
    --model_type hf \
    --model_name_or_path EleutherAI/pythia-70m \
    --dataset wikitext \
    --dataset_split "train[:5%]" \
    --num_samples 50000 \
    --output_dir data/metacognition \
    --batch_size 32
```

This will:
1. Load the specified model (HuggingFace or Mamba)
2. Process text from the specified dataset
3. For each token, extract hidden states and correctness labels
4. Save the data to the output directory, split into training and validation sets

## Step 2: Train the Metacognition Module

With the data prepared, you can now train the Metacognition module:

```bash
# Navigate to project root
cd /path/to/avian_cognition

# Train the metacognition module
python examples/train_metacognition.py \
    --train_data_path data/metacognition/train_metacognition_data.pt \
    --val_data_path data/metacognition/val_metacognition_data.pt \
    --hidden_dim 768 \  # Must match hidden_dim in the data
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --optimizer adamw \
    --early_stopping_patience 5 \
    --checkpoint_dir checkpoints/metacognition
```

This will:
1. Load the training and validation data
2. Create a MetacognitionModule with the specified hidden dimension
3. Train the module using the data
4. Save checkpoints and logs to the specified directory

## Step 3: Evaluate and Visualize

After training, you can evaluate the module's calibration on a test set:

```bash
# Navigate to project root
cd /path/to/avian_cognition

# Evaluate the trained module
python examples/evaluate_metacognition.py \
    --checkpoint_path checkpoints/metacognition/best_model.pt \
    --test_data_path data/metacognition/val_metacognition_data.pt \
    --visualize
```

This will:
1. Load the trained module from the checkpoint
2. Evaluate on the test data
3. Calculate metrics like Expected Calibration Error (ECE)
4. Generate reliability diagrams and other visualizations

## Integration with Main Model

Once trained, the Metacognition module can be integrated into the Avian Mamba model:

```python
from src.core.mamba_integration import AvianMambaModel
from src.modules.metacognition import MetacognitionModule

# Create model
model = AvianMambaModel(config)

# Load trained metacognition module
metacog_checkpoint = torch.load("checkpoints/metacognition/best_model.pt")
model.metacognition_module.load_state_dict(metacog_checkpoint["model_state_dict"])

# Now model can provide confidence estimates with predictions
```

## Hyperparameter Tuning

For best results, consider tuning:
- Learning rate
- Intermediate dimension
- Training duration
- Early stopping patience

## Tips for Best Performance

1. Use diverse data: The metacognition module will learn better if it sees a wide range of examples
2. Balance the dataset: Try to ensure a reasonable balance of correct/incorrect predictions
3. Use realistic data: Train on hidden states from the actual backbone model you'll be using
4. Evaluate using ECE: A well-calibrated model should have low Expected Calibration Error

## Troubleshooting

- If the model consistently predicts high or low confidence, try adjusting the class weights in the loss function
- If calibration is poor, check that the training data has a good balance of correct/incorrect examples
- If training is unstable, reduce the learning rate and add gradient clipping
