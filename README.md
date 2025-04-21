# Avian-Inspired Cognitive Architecture

## Convergent Intelligence Framework

A novel cognitive architecture that synthesizes state-space modeling (Mamba-SSM) with extreme quantization (BitNet) through the lens of avian neurobiology, creating a resource-efficient system capable of metacognition, Bayesian inference, planning, and numerical processing.

## Philosophical Foundation

This project explores intelligence as an emergent phenomenon decoupled from its physical substrate. Just as avian cognition evolved independently from mammalian intelligence—achieving comparable functions through radically different neural architectures—this framework demonstrates that sophisticated reasoning can emerge from computational structures fundamentally unlike conventional AI models.

## Core Components

- **Mamba-SSM Backbone**: Linear-time sequence modeling with selective state spaces
- **BitNet Quantization**: 1-bit weight representation for extreme efficiency
- **Metacognition Module**: Calibrated uncertainty estimation and self-monitoring
- **Bayesian Inference Module**: Probabilistic reasoning with belief state tracking
- **Planning/Tool-Use Module**: Multi-step reasoning and action sequence generation
- **Numerical Competence Module**: Precise arithmetic and numerical abstraction

## Implementation Roadmap

The project follows a phased approach to development:

1. **Core Mamba Model Setup**: Establish the backbone model and verify its performance
2. **Cognitive Module Integration**: Incrementally implement the four avian-inspired cognitive modules
3. **BitNet Quantization**: Ensure the entire system operates with 1-bit weight representation
4. **Training & Fine-Tuning**: Develop and apply specialized training protocols
5. **Evaluation & Benchmarking**: Rigorously validate the model's cognitive capabilities

## Training Infrastructure

The repository includes specialized training protocols for each cognitive module:

### Base Training Framework

The `training/base_trainer.py` provides a foundational training orchestrator with:

- Comprehensive experiment tracking with TensorBoard integration
- Checkpoint management for model state preservation
- Configurable training workflows with validation cycles
- Metrics visualization and analysis tools

### Specialized Cognitive Training

Each cognitive module has a dedicated training protocol:

- **Metacognition Training**: Calibrates the model's self-evaluation capabilities through systematic epistemological feedback
- **Bayesian Inference Training**: Shapes the model's belief-updating mechanisms through sequential evidence integration tasks
- **Planning Training**: Develops multi-step reasoning capabilities through structured problem-solving simulations
- **Numerical Training**: Cultivates precise arithmetic operations with extrapolation beyond training range

## Usage Examples

### Data Generation

```bash
# Generate metacognition training data from a pretrained language model
python scripts/generate_metacognition_data.py \
  --model_type hf \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset wikitext \
  --dataset_split "train[:5%]" \
  --num_samples 50000 \
  --output_dir data/metacognition
```

### Metacognition Training

```bash
# Train metacognition module with real data
python examples/train_metacognition.py \
  --train_data_path data/metacognition/train_metacognition_data.pt \
  --val_data_path data/metacognition/val_metacognition_data.pt \
  --hidden_dim 768 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --optimizer adamw \
  --early_stopping_patience 5 \
  --checkpoint_dir checkpoints/metacognition
```

### Metacognition Evaluation

```bash
# Evaluate trained metacognition module
python examples/evaluate_metacognition.py \
  --checkpoint_path checkpoints/metacognition/best_model.pt \
  --test_data_path data/metacognition/val_metacognition_data.pt \
  --visualize
```

### Integrated Model with Trained Modules

```bash
# Run the integrated model with trained cognitive modules
python examples/integrated_model.py \
  --model_size mini \
  --checkpoint_dir checkpoints \
  --visualize \
  --output_dir outputs
```

### Run Unit Tests

```bash
# Run unit tests for the metacognition module
python -m unittest tests/test_metacognition.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/angrysky56/avian_cognition.git
cd avian_cognition

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Project Structure

```
avian_cognition/
├── data/                 # Training data (generated)
│   └── metacognition/    # Metacognition training data
├── docs/                 # Documentation
│   ├── concepts.md       # Philosophical foundations
│   ├── architecture.md   # Technical architecture design
│   ├── roadmap.md        # Implementation roadmap
│   └── metacognition_training.md  # Training guide for metacognition
├── examples/             # Example scripts
│   ├── basic_example.py  # Simple demonstration of core components
│   ├── train_metacognition.py  # Metacognition training script
│   ├── train_bayesian.py  # Bayesian training script
│   ├── train_planning.py  # Planning training script
│   ├── train_numerical.py  # Numerical training script
│   ├── evaluate_metacognition.py  # Metacognition evaluation script
│   └── integrated_model.py  # Full model demonstration
├── scripts/              # Utility scripts
│   └── generate_metacognition_data.py  # Data generation for metacognition
├── src/                  # Source code
│   ├── core/             # Core architectural components
│   │   ├── bitnet.py     # BitNet quantization implementation
│   │   └── mamba_integration.py  # Mamba-SSM backbone integration
│   └── modules/          # Cognitive modules
│       ├── metacognition.py  # Metacognition implementation
│       ├── bayesian.py   # Bayesian inference implementation
│       ├── planning.py   # Planning implementation
│       └── numerical.py  # Numerical competence implementation
├── tests/                # Unit tests
│   └── test_metacognition.py  # Tests for metacognition module
└── training/             # Training protocols
    ├── base_trainer.py   # Base training framework
    ├── metacognition_trainer.py  # Metacognition training
    ├── bayesian_trainer.py  # Bayesian inference training
    ├── planning_trainer.py  # Planning training
    └── numerical_trainer.py  # Numerical competence training
```

## Current Status

The project is undergoing a major refactoring to transition from synthetic data to real training capabilities:

1. **Metacognition Module**: Now has complete training pipeline with:
   - Real data generation script using pretrained models
   - Proper dataset loading from model hidden states
   - Robust training with proper metrics and early stopping
   - Evaluation framework with calibration metrics

2. **Architectural Cleanup**: 
   - Removed synthetic datasets and demonstrations
   - Added proper docs explaining training procedures
   - Implemented unit tests to verify functionality

3. **Next Steps**:
   - Implement real data training for the Bayesian module
   - Develop training pipeline for Numerical competence
   - Create planning module training with real reasoning datasets

## Real Training Workflow

To train a cognitive module with real data:

```bash
# 1. Generate training data from a pretrained model
python scripts/generate_metacognition_data.py \
  --model_type hf \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset wikitext \
  --output_dir data/metacognition

# 2. Train the cognitive module
python examples/train_metacognition.py \
  --train_data_path data/metacognition/train_metacognition_data.pt \
  --val_data_path data/metacognition/val_metacognition_data.pt \
  --hidden_dim 768 \
  --learning_rate 1e-4 \
  --epochs 50

# 3. Evaluate the trained module
python examples/evaluate_metacognition.py \
  --checkpoint_path checkpoints/metacognition/best_model.pt \
  --test_data_path data/metacognition/val_metacognition_data.pt \
  --visualize
```

See the [Metacognition Training Guide](docs/metacognition_training.md) for detailed instructions.

## License

MIT License