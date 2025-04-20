"""
Base Trainer Framework

This module defines the foundational training protocol abstractions,
establishing the temporal rhythms through which cognitive emergence transpires.
"""

import os
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CognitiveTrainer(ABC):
    """
    Abstract orchestrator of neural evolution processes.
    
    Establishes the fundamental cadence through which cognitive modules
    transcend their initialized state into coherent functional entities,
    cultivating capabilities through iterative parameter refinement.
    
    This temporal scaffold provides the rhythmic structure within which
    specialized cognitive faculties—metacognition, Bayesian inference,
    planning, and numerical processing—may emerge and crystallize.
    
    Attributes:
        model: The neural architecture undergoing evolution
        optimizer: The mechanism governing parameter trajectory
        criterion: The evaluative lens through which progress is measured
        device: The material substrate upon which computation manifests
        config: The hyperparameter constellation guiding the process
        writer: The observer capturing evolutionary metrics
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        experiment_name=None
    ):
        """
        Initialize the evolutionary orchestrator.
        
        Args:
            model: Neural architecture to evolve
            optimizer: Parameter update mechanism
            criterion: Loss evaluation function
            device: Computational substrate
            config: Hyperparameter configuration
            experiment_name: Optional identifier for this evolutionary sequence
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Create experiment identifier
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = model.__class__.__name__
            experiment_name = f"{model_type}_{timestamp}"
            
        self.experiment_name = experiment_name
        
        # Establish logging
        self.log_dir = os.path.join("logs", experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize evolution observer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        # Track evolutionary trajectory
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
    @abstractmethod
    def train_epoch(self, train_loader):
        """
        Conduct a single evolutionary cycle over the training ecosystem.
        
        Args:
            train_loader: Iterator through training examples
            
        Returns:
            metrics: Quantified aspects of evolutionary progress
        """
        pass
        
    @abstractmethod
    def validate(self, val_loader):
        """
        Evaluate current developmental state on validation ecosystem.
        
        Args:
            val_loader: Iterator through validation examples
            
        Returns:
            metrics: Quantified aspects of evolutionary fitness
        """
        pass
        
    def train(self, train_loader, val_loader, epochs):
        """
        Orchestrate the complete evolutionary sequence.
        
        Args:
            train_loader: Training data ecosystem
            val_loader: Validation data ecosystem
            epochs: Total developmental cycles
            
        Returns:
            trajectory: Historical record of evolutionary metrics
        """
        self.logger.info(f"Beginning evolutionary sequence: {self.experiment_name}")
        self.logger.info(f"Model architecture: {self.model.__class__.__name__}")
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        
        trajectory = {
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Main evolutionary loop
        for epoch in range(epochs):
            self.epoch = epoch
            self.logger.info(f"Initiating epoch {epoch}/{epochs-1}")
            
            # Training cycle
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - start_time
            
            # Log training progress
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s - Metrics: {train_metrics}")
            for metric_name, metric_value in train_metrics.items():
                self.writer.add_scalar(f"train/{metric_name}", metric_value, self.epoch)
            
            # Validation cycle
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Log validation results
                self.logger.info(f"Validation results - Metrics: {val_metrics}")
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f"val/{metric_name}", metric_value, self.epoch)
                
                # Track best-performing state
                if val_metrics.get('loss', float('inf')) < self.best_metric:
                    self.best_metric = val_metrics.get('loss', float('inf'))
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best model saved - Validation loss: {self.best_metric:.6f}")
            
            # Regular state preservation
            if epoch % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(is_best=False)
            
            # Accumulate metrics
            trajectory['train_metrics'].append(train_metrics)
            if val_loader is not None:
                trajectory['val_metrics'].append(val_metrics)
        
        self.logger.info(f"Evolutionary sequence complete - {epochs} epochs")
        self.writer.close()
        
        return trajectory
        
    def save_checkpoint(self, is_best=False):
        """
        Preserve the current developmental state.
        
        Args:
            is_best: Whether this state represents peak fitness
        """
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare state record
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if indicated
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path):
        """
        Restore a previous developmental state.
        
        Args:
            checkpoint_path: Path to preserved state record
        """
        self.logger.info(f"Restoring evolutionary state from {checkpoint_path}")
        
        # Load state record
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore evolutionary tracking
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"Restored to epoch {self.epoch}, step {self.global_step}")
