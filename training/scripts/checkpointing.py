#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpointing utility script for Mistral-7B fine-tuning.
This script provides utilities for saving and loading model checkpoints,
which is especially important for Google Colab as it may disconnect.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training/logs/checkpointing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


class CheckpointManager:
    """Manager for saving and loading checkpoints."""

    def __init__(self, base_dir: str = "training/models", max_checkpoints: int = 3):
        """Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.base_dir = base_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info(f"Initialized checkpoint manager with base directory: {base_dir}")
        logger.info(f"Maximum checkpoints to keep: {max_checkpoints}")

    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch: int = 0,
        global_step: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """Save a model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch
            global_step: Current global step
            loss: Current loss value
            metrics: Current evaluation metrics
            checkpoint_name: Optional name for the checkpoint

        Returns:
            Path to the saved checkpoint
        """
        # Generate checkpoint name if not provided
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_step{global_step}_{timestamp}"

        # Create the checkpoint directory
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_path}")

        # Save model
        if hasattr(model, "save_pretrained"):
            # For transformers models
            model.save_pretrained(checkpoint_path)
        else:
            # For regular PyTorch models
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))

        # Save optimizer state
        if optimizer is not None:
            torch.save(
                optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt")
            )

        # Save scheduler state
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt")
            )

        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_name}")
        return checkpoint_path

    def load_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load a model checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            scheduler: Scheduler to load into
            checkpoint_path: Path to the checkpoint directory
            device: Device to load model to

        Returns:
            Dictionary of training state
        """
        # If no specific checkpoint path is provided, use the latest
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            logger.warning("No checkpoint found to load")
            return {}

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model
        if hasattr(model, "from_pretrained"):
            # For transformers models
            model = model.from_pretrained(checkpoint_path, device_map="auto")
        else:
            # For regular PyTorch models
            model_path = os.path.join(checkpoint_path, "model.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))

        # Load optimizer state
        if optimizer is not None:
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device)
                )

        # Load scheduler state
        if scheduler is not None:
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler.load_state_dict(torch.load(scheduler_path))

        # Load training state
        training_state = {}
        training_state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)

        logger.info(
            f"Checkpoint loaded. Epoch: {training_state.get('epoch', 0)}, Step: {training_state.get('global_step', 0)}"
        )
        return training_state

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns:
            Path to the latest checkpoint or None if no checkpoints found
        """
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None

        latest_checkpoint = checkpoints[-1]
        return os.path.join(self.checkpoint_dir, latest_checkpoint)

    def _list_checkpoints(self) -> List[str]:
        """List all checkpoints in order of creation.

        Returns:
            List of checkpoint names
        """
        if not os.path.exists(self.checkpoint_dir):
            return []

        # Get all subdirectories
        checkpoints = [
            d
            for d in os.listdir(self.checkpoint_dir)
            if os.path.isdir(os.path.join(self.checkpoint_dir, d))
        ]

        # Sort by creation time
        checkpoints.sort(
            key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x))
        )

        return checkpoints

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the maximum number."""
        checkpoints = self._list_checkpoints()

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Remove the oldest checkpoints
        num_to_remove = len(checkpoints) - self.max_checkpoints
        for i in range(num_to_remove):
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[i])
            logger.info(f"Removing old checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path, ignore_errors=True)

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their metadata.

        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = self._list_checkpoints()
        checkpoint_info = []

        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
            training_state_path = os.path.join(checkpoint_path, "training_state.json")

            info = {
                "name": checkpoint,
                "path": checkpoint_path,
                "created": datetime.fromtimestamp(
                    os.path.getctime(checkpoint_path)
                ).isoformat(),
            }

            if os.path.exists(training_state_path):
                with open(training_state_path, "r") as f:
                    training_state = json.load(f)
                    info.update(
                        {
                            "epoch": training_state.get("epoch", 0),
                            "global_step": training_state.get("global_step", 0),
                            "loss": training_state.get("loss", 0.0),
                            "metrics": training_state.get("metrics", {}),
                        }
                    )

            checkpoint_info.append(info)

        return checkpoint_info

    def print_checkpoint_summary(self) -> None:
        """Print a summary of available checkpoints."""
        checkpoints = self.list_available_checkpoints()

        if not checkpoints:
            logger.info("No checkpoints available")
            return

        logger.info(f"Found {len(checkpoints)} checkpoints:")

        for i, checkpoint in enumerate(checkpoints):
            logger.info(f"{i+1}. {checkpoint['name']}")
            logger.info(f"   - Created: {checkpoint.get('created', 'unknown')}")
            logger.info(
                f"   - Epoch: {checkpoint.get('epoch', 'unknown')}, Step: {checkpoint.get('global_step', 'unknown')}"
            )

            if "loss" in checkpoint:
                logger.info(f"   - Loss: {checkpoint['loss']:.4f}")

            metrics = checkpoint.get("metrics", {})
            if metrics:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"   - Metrics: {metrics_str}")

            logger.info("")


# Utility functions for Google Colab integration
def save_to_google_drive(source_dir: str, target_dir: str) -> bool:
    """Save files to Google Drive for persistence.

    Args:
        source_dir: Source directory in local storage
        target_dir: Target directory in Google Drive

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Copy files
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(target_dir, item)

            if os.path.isdir(s):
                # Recursively copy directories
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                # Copy files
                shutil.copy2(s, d)

        logger.info(f"Successfully saved files from {source_dir} to {target_dir}")
        return True

    except Exception as e:
        logger.error(f"Error saving to Google Drive: {e}")
        return False


def load_from_google_drive(source_dir: str, target_dir: str) -> bool:
    """Load files from Google Drive to local storage.

    Args:
        source_dir: Source directory in Google Drive
        target_dir: Target directory in local storage

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Copy files
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(target_dir, item)

            if os.path.isdir(s):
                # Recursively copy directories
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                # Copy files
                shutil.copy2(s, d)

        logger.info(f"Successfully loaded files from {source_dir} to {target_dir}")
        return True

    except Exception as e:
        logger.error(f"Error loading from Google Drive: {e}")
        return False
