"""
Planning Module Training Protocol

This module implements the specialized neural cultivation process for the
planning circuit, enabling multi-step reasoning and prospective cognition
through temporal abstraction and causal modeling.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from .base_trainer import CognitiveTrainer


class PlanningTrainer(CognitiveTrainer):
    """
    Specialized evolutionary orchestrator for sequential reasoning.
    
    Cultivates the system's capacity for multi-step planning through a
    developmental protocol that shapes causal understanding and temporal
    abstraction, mirroring the tool-use and prospective cognition
    capabilities observed in corvids.
    
    The training rhythm structures the learning process through increasing
    levels of reasoning complexity, fostering the emergence of abstract
    planning schemas that can be applied to novel problem domains.
    
    Attributes:
        model: Planning module or integrated architecture
        backbone: Optional underlying model for generating representations
        planning_steps: Number of reasoning steps to generate
        difficulty_curriculum: Progressive complexity scaling
        step_accuracy_weight: Importance of intermediate step correctness
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        backbone=None,
        planning_steps=5,
        difficulty_curriculum=None,
        step_accuracy_weight=0.5,
        experiment_name="PlanningTraining"
    ):
        """
        Initialize the planning training protocol.
        
        Args:
            model: Neural architecture to evolve
            optimizer: Parameter update mechanism
            criterion: Loss evaluation function
            device: Computational substrate
            config: Hyperparameter configuration
            backbone: Optional base model for generating representations
            planning_steps: Number of reasoning steps to generate
            difficulty_curriculum: Progressive complexity scaling
            step_accuracy_weight: Importance of intermediate step correctness
            experiment_name: Identifier for this evolutionary sequence
        """
        super().__init__(model, optimizer, criterion, device, config, experiment_name)
        
        self.backbone = backbone
        self.planning_steps = planning_steps
        
        # Set default difficulty curriculum if not provided
        self.difficulty_curriculum = difficulty_curriculum if difficulty_curriculum is not None else {
            0: 1,    # Initial epoch: 1-step problems
            5: 2,    # Epoch 5: 2-step problems
            10: 3,   # Epoch 10: 3-step problems
            15: 4,   # Epoch 15: 4-step problems
            20: 5    # Epoch 20: 5-step problems
        }
        
        self.step_accuracy_weight = step_accuracy_weight
        
        # Track reasoning progression
        self.solution_accuracy_history = []
        self.step_accuracy_history = []
        self.step_importance_history = []
        
    def _extract_planning_module(self, model):
        """
        Isolate planning circuitry from integrated architecture.
        
        Many architectural variants may contain planning components,
        this function navigates the architectural hierarchy to locate
        the specific circuitry responsible for sequential reasoning.
        
        Args:
            model: Integrated neural architecture
            
        Returns:
            planning_module: Isolated planning circuit
        """
        # Direct access if model is the planning module itself
        if hasattr(model, 'plan_cell'):
            return model
            
        # Extract from AvianMamba architecture
        if hasattr(model, 'planning_module'):
            return model.planning_module
            
        # Find module by name in children
        for name, child in model.named_children():
            if 'plan' in name.lower() and hasattr(child, 'plan_cell'):
                return child
                
        # Default to the provided model
        return model
        
    def _generate_reasoning_task(self, batch_size, num_steps):
        """
        Generate synthetic multi-step reasoning problems.
        
        Creates structured reasoning tasks with ground-truth intermediate
        steps and final solutions, enabling training of the planning
        module on sequential problem solving.
        
        This generator creates abstract reasoning problems that test
        the model's ability to chain multiple mental operations.
        
        Args:
            batch_size: Number of problems to generate
            num_steps: Complexity of reasoning tasks (number of steps)
            
        Returns:
            problem_states: Problem context representations
            step_sequences: Ground-truth reasoning steps
            solutions: Correct final solutions
            contexts: Optional context sequences for attention
        """
        hidden_dim = self.config.get('hidden_dim', 256)
        
        # Create problem representations
        problem_states = torch.zeros(batch_size, hidden_dim, device=self.device)
        
        # Create context sequences (for attention)
        seq_len = 5  # Fixed length for simplicity
        contexts = torch.zeros(batch_size, seq_len, hidden_dim, device=self.device)
        
        # Initialize storage for ground-truth steps and solutions
        step_sequences = torch.zeros(num_steps, batch_size, hidden_dim, device=self.device)
        solutions = torch.zeros(batch_size, hidden_dim, device=self.device)
        
        # Lambda for creating a random unit vector
        random_unit_vector = lambda: F.normalize(torch.randn(hidden_dim, device=self.device), dim=0)
        
        # Task types (abstract reasoning patterns)
        task_types = [
            'sequential_transformation',  # Apply transforms in sequence
            'recursive_pattern',          # Apply same transform repeatedly
            'hierarchical_grouping',      # Group and then transform
            'conditional_sequence',       # If-then branching logic
            'compositional_integration'   # Combine partial results
        ]
        
        # Generate each problem
        for i in range(batch_size):
            # Select random task type
            task_type = np.random.choice(task_types)
            
            # Encode task type in problem state
            task_idx = task_types.index(task_type)
            problem_states[i, task_idx] = 1.0
            
            # Create reference vectors for this problem
            base_vector = random_unit_vector()
            transform_vectors = [random_unit_vector() for _ in range(num_steps)]
            
            # Initialize context with task-relevant information
            for j in range(seq_len):
                # Encode task-specific context
                if j == 0:
                    contexts[i, j] = base_vector
                elif j < seq_len - 1:
                    contexts[i, j] = transform_vectors[min(j-1, num_steps-1)]
            
            # Generate ground-truth reasoning steps based on task type
            if task_type == 'sequential_transformation':
                # Apply sequence of different transformations
                current_state = base_vector
                for step in range(num_steps):
                    # Apply transform
                    current_state = 0.8 * current_state + 0.2 * transform_vectors[step]
                    current_state = F.normalize(current_state, dim=0)
                    
                    # Store step
                    step_sequences[step, i] = current_state
                
                # Final solution is the last state
                solutions[i] = current_state
                
            elif task_type == 'recursive_pattern':
                # Apply same transformation repeatedly
                transform = transform_vectors[0]
                current_state = base_vector
                
                for step in range(num_steps):
                    # Apply transform
                    current_state = 0.7 * current_state + 0.3 * transform
                    current_state = F.normalize(current_state, dim=0)
                    
                    # Store step
                    step_sequences[step, i] = current_state
                
                # Final solution is the last state
                solutions[i] = current_state
                
            elif task_type == 'hierarchical_grouping':
                # Group then transform
                groups = min(num_steps - 1, 2)  # At least 1 transform after grouping
                
                # Grouping steps
                current_state = base_vector
                for step in range(groups):
                    # Combine with transform
                    current_state = 0.6 * current_state + 0.4 * transform_vectors[step]
                    current_state = F.normalize(current_state, dim=0)
                    
                    # Store step
                    step_sequences[step, i] = current_state
                
                # Transformation steps
                for step in range(groups, num_steps):
                    # Apply final transform
                    current_state = 0.5 * current_state + 0.5 * transform_vectors[groups]
                    current_state = F.normalize(current_state, dim=0)
                    
                    # Store step
                    step_sequences[step, i] = current_state
                
                # Final solution is the last state
                solutions[i] = current_state
                
            elif task_type == 'conditional_sequence':
                # If-then branching logic based on initial condition
                branch_condition = (base_vector[0] > 0)  # Arbitrary condition
                
                current_state = base_vector
                for step in range(num_steps):
                    # Apply different transform based on branch
                    if branch_condition:
                        transform_idx = min(step, len(transform_vectors) - 1)
                    else:
                        transform_idx = min(num_steps - 1 - step, len(transform_vectors) - 1)
                        
                    transform = transform_vectors[transform_idx]
                    
                    # Apply transform
                    current_state = 0.65 * current_state + 0.35 * transform
                    current_state = F.normalize(current_state, dim=0)
                    
                    # Store step
                    step_sequences[step, i] = current_state
                
                # Final solution is the last state
                solutions[i] = current_state
                
            elif task_type == 'compositional_integration':
                # Combine partial results
                partial_results = [base_vector]
                
                # Generate partial results
                for step in range(num_steps - 1):
                    partial = 0.6 * base_vector + 0.4 * transform_vectors[step]
                    partial = F.normalize(partial, dim=0)
                    partial_results.append(partial)
                    
                    # Store step
                    step_sequences[step, i] = partial
                
                # Final step integrates all partial results
                integration = sum(partial_results) / len(partial_results)
                integration = F.normalize(integration, dim=0)
                
                # Store final step
                step_sequences[num_steps - 1, i] = integration
                
                # Final solution
                solutions[i] = integration
        
        return problem_states, step_sequences, solutions, contexts
        
    def _evaluate_step_similarity(self, predicted_steps, ground_truth_steps):
        """
        Evaluate similarity between predicted and ground-truth reasoning steps.
        
        Args:
            predicted_steps: Model's predicted reasoning steps
            ground_truth_steps: Ground-truth reasoning steps
            
        Returns:
            step_similarities: Cosine similarity for each step
            overall_similarity: Average similarity across steps
        """
        # Ensure correct dimensions
        num_steps, batch_size, _ = ground_truth_steps.shape
        
        # Calculate cosine similarity for each step
        step_similarities = []
        
        for step in range(num_steps):
            # Get vectors for this step
            pred = predicted_steps[step]
            truth = ground_truth_steps[step]
            
            # Normalize
            pred_norm = F.normalize(pred, dim=1)
            truth_norm = F.normalize(truth, dim=1)
            
            # Calculate cosine similarity
            similarity = (pred_norm * truth_norm).sum(dim=1)
            
            # Store batch-averaged similarity
            step_similarities.append(similarity.mean().item())
        
        # Calculate overall similarity
        overall_similarity = sum(step_similarities) / len(step_similarities)
        
        return step_similarities, overall_similarity
        
    def _evaluate_solution_similarity(self, predicted_solution, ground_truth_solution):
        """
        Evaluate similarity between predicted and ground-truth final solutions.
        
        Args:
            predicted_solution: Model's predicted solution
            ground_truth_solution: Ground-truth solution
            
        Returns:
            solution_similarity: Cosine similarity of solutions
        """
        # Normalize
        pred_norm = F.normalize(predicted_solution, dim=1)
        truth_norm = F.normalize(ground_truth_solution, dim=1)
        
        # Calculate cosine similarity
        similarity = (pred_norm * truth_norm).sum(dim=1)
        
        # Return batch-averaged similarity
        return similarity.mean().item()
        
    def train_epoch(self, train_loader=None):
        """
        Conduct a single evolutionary cycle for planning capability.
        
        This cycle orchestrates the refinement of sequential reasoning
        through progressive complexity scaling, developing the module's
        capacity for multi-step planning and causal understanding.
        
        Args:
            train_loader: Optional iterator through training examples
                         (if None, generate synthetic data)
            
        Returns:
            metrics: Quantified aspects of planning capability
        """
        # Set to training mode
        self.model.train()
        if self.backbone is not None:
            self.backbone.eval()  # Freeze backbone during planning training
            
        # Isolate planning module if needed
        planning_module = self._extract_planning_module(self.model)
        
        # Determine problem complexity based on curriculum
        num_steps = 1  # Default to simplest problems
        for epoch_threshold, steps in sorted(self.difficulty_curriculum.items()):
            if self.epoch >= epoch_threshold:
                num_steps = steps
        
        # Tracking metrics
        total_loss = 0
        total_samples = 0
        total_solution_similarity = 0
        total_step_similarity = 0
        step_similarities = []
        step_importances = []
        
        # Determine number of batches
        if train_loader is not None:
            num_batches = len(train_loader)
        else:
            # Default number of synthetic batches
            num_batches = self.config.get('synthetic_batches', 100)
            
        # Batch size
        batch_size = self.config.get('batch_size', 32)
        
        # Evolutionary iteration
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {self.epoch} (Steps: {num_steps})"):
            # Either use provided data or generate synthetic problems
            if train_loader is not None:
                batch = next(iter(train_loader))
                problem_states = batch['problem_states'].to(self.device)
                ground_truth_steps = batch['step_sequences'].to(self.device)
                ground_truth_solutions = batch['solutions'].to(self.device)
                contexts = batch.get('contexts', None)
                if contexts is not None:
                    contexts = contexts.to(self.device)
            else:
                # Generate synthetic reasoning tasks
                problem_states, ground_truth_steps, ground_truth_solutions, contexts = self._generate_reasoning_task(
                    batch_size, num_steps
                )
            
            # Reset gradient pathways
            self.optimizer.zero_grad()
            
            # Forward pass through planning module
            plan_embedding, step_states, step_importances_batch = planning_module(
                problem_states, contexts
            )
            
            # Calculate solution similarity loss
            solution_similarity = self._evaluate_solution_similarity(plan_embedding, ground_truth_solutions)
            solution_loss = 1.0 - solution_similarity
            
            # Calculate step similarity loss
            step_similarity_list, overall_step_similarity = self._evaluate_step_similarity(
                step_states, ground_truth_steps[:num_steps]
            )
            step_loss = 1.0 - overall_step_similarity
            
            # Combined loss with weighting
            loss = (1.0 - self.step_accuracy_weight) * solution_loss + self.step_accuracy_weight * step_loss
            
            # Backward propagation of alignment signal
            loss.backward()
            
            # Update parameters along improvement gradient
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = problem_states.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_solution_similarity += solution_similarity * batch_size
            total_step_similarity += overall_step_similarity * batch_size
            
            # Accumulate step-wise metrics
            step_similarities.append(step_similarity_list)
            step_importances.append(step_importances_batch.detach().cpu().numpy())
            
            # Update step counter
            self.global_step += 1
            
        # Calculate epoch-level metrics
        avg_loss = total_loss / total_samples
        avg_solution_similarity = total_solution_similarity / total_samples
        avg_step_similarity = total_step_similarity / total_samples
        
        # Calculate average step similarities and importances
        avg_step_similarities = []
        if step_similarities:
            # Transpose to get list per step
            step_similarities_per_step = list(zip(*step_similarities))
            avg_step_similarities = [sum(s) / len(s) for s in step_similarities_per_step]
        
        avg_step_importances = []
        if step_importances:
            step_importances_array = np.concatenate(step_importances, axis=0)
            avg_step_importances = np.mean(step_importances_array, axis=0).tolist()
        
        # Record history
        self.solution_accuracy_history.append(avg_solution_similarity)
        self.step_accuracy_history.append(avg_step_similarity)
        self.step_importance_history.append(avg_step_importances)
        
        # Generate visualizations
        if self.epoch % self.config.get('plot_interval', 5) == 0:
            self._plot_step_accuracy()
            self._plot_step_importance()
            
        # Return consolidated metrics
        metrics = {
            'loss': avg_loss,
            'solution_similarity': avg_solution_similarity,
            'step_similarity': avg_step_similarity,
            'num_steps': num_steps
        }
        
        # Add per-step metrics
        for i, similarity in enumerate(avg_step_similarities):
            metrics[f'step_{i}_similarity'] = similarity
            
        for i, importance in enumerate(avg_step_importances):
            metrics[f'step_{i}_importance'] = importance
        
        return metrics
        
    def validate(self, val_loader=None):
        """
        Evaluate planning capability on validation tasks.
        
        Tests the model's ability to generate coherent reasoning steps
        and arrive at correct solutions for novel problems.
        
        Args:
            val_loader: Optional iterator through validation examples
                       (if None, generate synthetic data)
            
        Returns:
            metrics: Quantified aspects of planning capability
        """
        # Set to evaluation mode
        self.model.eval()
        if self.backbone is not None:
            self.backbone.eval()
            
        # Isolate planning module if needed
        planning_module = self._extract_planning_module(self.model)
        
        # Determine problem complexity based on curriculum
        num_steps = max(self.difficulty_curriculum.values())  # Use max complexity for evaluation
        
        # Tracking metrics
        total_loss = 0
        total_samples = 0
        total_solution_similarity = 0
        total_step_similarity = 0
        
        # Store step sequences for visualization
        all_step_states = []
        all_ground_truth_steps = []
        
        # Determine number of batches
        if val_loader is not None:
            num_batches = min(len(val_loader), self.config.get('val_batches', 10))
        else:
            # Default number of synthetic batches
            num_batches = self.config.get('val_batches', 10)
            
        # Batch size
        batch_size = self.config.get('batch_size', 32)
        
        # Evaluation without gradient tracking
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Validation"):
                # Either use provided data or generate synthetic problems
                if val_loader is not None:
                    batch = next(iter(val_loader))
                    problem_states = batch['problem_states'].to(self.device)
                    ground_truth_steps = batch['step_sequences'].to(self.device)
                    ground_truth_solutions = batch['solutions'].to(self.device)
                    contexts = batch.get('contexts', None)
                    if contexts is not None:
                        contexts = contexts.to(self.device)
                else:
                    # Generate synthetic reasoning tasks
                    problem_states, ground_truth_steps, ground_truth_solutions, contexts = self._generate_reasoning_task(
                        batch_size, num_steps
                    )
                
                # Forward pass through planning module
                plan_embedding, step_states, step_importances = planning_module(
                    problem_states, contexts
                )
                
                # Calculate solution similarity
                solution_similarity = self._evaluate_solution_similarity(plan_embedding, ground_truth_solutions)
                
                # Calculate step similarity
                _, overall_step_similarity = self._evaluate_step_similarity(
                    step_states, ground_truth_steps[:num_steps]
                )
                
                # Combined loss with weighting
                loss = (1.0 - self.step_accuracy_weight) * (1.0 - solution_similarity) + \
                      self.step_accuracy_weight * (1.0 - overall_step_similarity)
                
                # Accumulate metrics
                batch_size = problem_states.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                total_solution_similarity += solution_similarity * batch_size
                total_step_similarity += overall_step_similarity * batch_size
                
                # Store step sequences for first batch only
                if batch_idx == 0:
                    all_step_states = step_states.detach().cpu().numpy()
                    all_ground_truth_steps = ground_truth_steps[:num_steps].detach().cpu().numpy()
            
            # Calculate validation metrics
            avg_loss = total_loss / total_samples
            avg_solution_similarity = total_solution_similarity / total_samples
            avg_step_similarity = total_step_similarity / total_samples
        
        # Generate validation visualizations
        if len(all_step_states) > 0:
            self._plot_step_trajectory(
                all_step_states, all_ground_truth_steps,
                title=f"Validation Step Trajectory (Epoch {self.epoch})",
                filename=f"val_step_trajectory_epoch_{self.epoch}.png"
            )
            
        # Return validation metrics
        metrics = {
            'val_loss': avg_loss,
            'val_solution_similarity': avg_solution_similarity,
            'val_step_similarity': avg_step_similarity,
            'val_num_steps': num_steps
        }
        
        return metrics
        
    def _plot_step_accuracy(self):
        """
        Visualize accuracy of intermediate reasoning steps.
        
        Creates a plot showing how well the planning module generates
        each step in the reasoning sequence compared to ground truth.
        """
        if not self.step_accuracy_history:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        epochs = range(len(self.step_accuracy_history))
        
        # Plot overall step accuracy
        ax.plot(epochs, self.step_accuracy_history, 'k-', linewidth=2, label='Overall Step Similarity')
        
        # Plot solution accuracy
        ax.plot(epochs, self.solution_accuracy_history, 'b-', linewidth=2, label='Solution Similarity')
        
        # Configure plot
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Planning Step Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save figure
        save_path = f"{self.log_dir}/plots/step_accuracy_epoch_{self.epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "planning/step_accuracy",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
    def _plot_step_importance(self):
        """
        Visualize importance weights assigned to reasoning steps.
        
        Creates a plot showing how the planning module weights the
        importance of each step in the reasoning sequence.
        """
        if not self.step_importance_history or not self.step_importance_history[-1]:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get latest importance weights
        importances = self.step_importance_history[-1]
        
        # Plot importance weights
        steps = range(len(importances))
        ax.bar(steps, importances, alpha=0.7)
        
        # Configure plot
        ax.set_xlabel('Reasoning Step')
        ax.set_ylabel('Importance Weight')
        ax.set_title(f'Planning Step Importance (Epoch {self.epoch})')
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save figure
        save_path = f"{self.log_dir}/plots/step_importance_epoch_{self.epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "planning/step_importance",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
    def _plot_step_trajectory(self, step_states, ground_truth_steps, title=None, filename=None):
        """
        Visualize reasoning step trajectory in latent space.
        
        Creates a 2D visualization of the model's reasoning steps compared
        to ground truth, showing how the planning module navigates through
        the solution space.
        
        Args:
            step_states: Model's predicted reasoning steps
            ground_truth_steps: Ground-truth reasoning steps
            title: Optional title for the plot
            filename: Optional filename for saving the visualization
        """
        try:
            # Apply PCA to reduce dimensionality for visualization
            pca = PCA(n_components=2)
            
            # Reshape to combine batch and step dimensions
            num_steps, batch_size, hidden_dim = step_states.shape
            
            # Select single example for clarity (first in batch)
            example_idx = 0
            predicted_steps = step_states[:, example_idx, :]
            truth_steps = ground_truth_steps[:, example_idx, :]
            
            # Combine for PCA
            combined = np.vstack([predicted_steps, truth_steps])
            
            # Apply PCA
            reduced = pca.fit_transform(combined)
            
            # Split back
            pred_reduced = reduced[:num_steps]
            truth_reduced = reduced[num_steps:]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot predicted trajectory
            ax.plot(pred_reduced[:, 0], pred_reduced[:, 1], 'b-', marker='o', linewidth=2,
                   label='Predicted Steps')
                   
            # Plot ground truth trajectory
            ax.plot(truth_reduced[:, 0], truth_reduced[:, 1], 'r--', marker='x', linewidth=2,
                   label='Ground Truth Steps')
                   
            # Mark steps
            for i, (x, y) in enumerate(pred_reduced):
                ax.text(x, y, f"P{i}", fontsize=12, ha='right')
                
            for i, (x, y) in enumerate(truth_reduced):
                ax.text(x, y, f"T{i}", fontsize=12, ha='left')
            
            # Configure plot
            if title:
                ax.set_title(title)
            else:
                ax.set_title('Reasoning Step Trajectory in Latent Space')
                
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add variance explanation
            var_ratio = pca.explained_variance_ratio_
            ax.text(0.02, 0.98, f"PCA variance: {var_ratio[0]:.2f}, {var_ratio[1]:.2f}",
                   transform=ax.transAxes, fontsize=10, va='top',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Save figure
            if filename:
                save_path = f"{self.log_dir}/plots/{filename}"
            else:
                save_path = f"{self.log_dir}/plots/step_trajectory_epoch_{self.epoch}.png"
                
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            
            # Log to tensorboard
            self.writer.add_figure(
                "planning/step_trajectory",
                fig,
                global_step=self.epoch
            )
            
            # Close figure to prevent memory leaks
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"Error generating step trajectory plot: {e}")
        
    def plot_training_trajectory(self):
        """
        Visualize the evolutionary trajectory of planning capability.
        
        Creates a comprehensive view of the system's development of
        sequential reasoning capabilities over the training process.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Accuracy progression
        ax1 = axes[0]
        
        # Prepare data
        epochs = range(len(self.solution_accuracy_history))
        
        # Plot solution and step accuracy
        ax1.plot(epochs, self.solution_accuracy_history, 'b-', linewidth=2, label='Solution Similarity')
        ax1.plot(epochs, self.step_accuracy_history, 'g-', linewidth=2, label='Step Similarity')
        
        # Plot target complexity progression
        complexity = []
        for epoch in epochs:
            steps = 1  # Default
            for e_threshold, s in sorted(self.difficulty_curriculum.items()):
                if epoch >= e_threshold:
                    steps = s
            complexity.append(steps / max(self.difficulty_curriculum.values()))
            
        ax1.plot(epochs, complexity, 'r--', linewidth=1.5, label='Problem Complexity')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Similarity / Complexity')
        ax1.set_title('Planning Accuracy Evolution')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Step importance evolution
        ax2 = axes[1]
        
        # Prepare importance data
        if self.step_importance_history:
            # Get step importance at specific epochs
            max_steps = max(len(imp) for imp in self.step_importance_history if imp)
            
            if max_steps > 0:
                # Extract epochs at regular intervals
                num_samples = min(5, len(self.step_importance_history))
                sample_epochs = [epochs[int(i * len(epochs) / num_samples)] for i in range(num_samples)]
                
                # Plot importance for each sampled epoch
                for i, epoch in enumerate(sample_epochs):
                    if epoch < len(self.step_importance_history):
                        imp = self.step_importance_history[epoch]
                        if imp and len(imp) > 0:
                            # Pad with zeros if needed
                            padded_imp = imp + [0] * (max_steps - len(imp))
                            ax2.plot(range(max_steps), padded_imp, marker='o',
                                   label=f'Epoch {epoch}')
                
                ax2.set_xlabel('Reasoning Step')
                ax2.set_ylabel('Importance Weight')
                ax2.set_title('Step Importance Evolution')
                ax2.set_xticks(range(max_steps))
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = f"{self.log_dir}/plots/planning_training_trajectory.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "training/planning_trajectory",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
