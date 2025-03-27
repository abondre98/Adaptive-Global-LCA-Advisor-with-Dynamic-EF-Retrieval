#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab integration script for Mistral-7B fine-tuning.
This script provides utilities for setting up Google Colab
and handling Google Drive integration for the fine-tuning process.
"""

import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from IPython.display import HTML, display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training/logs/colab_integration.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


def is_colab_runtime() -> bool:
    """Check if we are running in a Google Colab runtime.

    Returns:
        True if running in Google Colab, False otherwise
    """
    try:
        import google.colab

        return True
    except ImportError:
        return False


def install_requirements() -> None:
    """Install required packages for fine-tuning."""

    requirements = [
        "torch>=2.0.0",
        "transformers>=4.34.0",
        "peft>=0.5.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.40.0",
        "trl>=0.7.1",
        "tensorboard>=2.14.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "tqdm>=4.66.1",
        "pandas>=2.1.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "sentencepiece>=0.1.99",
        "scipy>=1.11.2",
        "scikit-learn>=1.3.0",
        "einops>=0.6.1",
        "wandb>=0.15.10",
    ]

    logger.info("Installing requirements...")

    for package in requirements:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package]
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")

    # Special handling for flash-attn
    try:
        logger.info("Installing flash-attn (if compatible with the environment)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "flash-attn"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "Could not install flash-attn. This is optional and may not be compatible with all environments."
        )

    logger.info("Requirements installation complete")


def mount_google_drive() -> str:
    """Mount Google Drive and return the path to the mount point.

    Returns:
        Path to the mounted Google Drive
    """
    if not is_colab_runtime():
        logger.warning("Not running in Google Colab, cannot mount Google Drive")
        return ""

    logger.info("Mounting Google Drive...")

    from google.colab import drive

    drive.mount("/content/drive")

    logger.info("Google Drive mounted at /content/drive")
    return "/content/drive/MyDrive"


def setup_project_structure(base_dir: str = "training") -> Dict[str, str]:
    """Set up the project directory structure.

    Args:
        base_dir: Base directory for the project

    Returns:
        Dictionary of directory paths
    """
    directories = {
        "base": base_dir,
        "data": os.path.join(base_dir, "data"),
        "models": os.path.join(base_dir, "models"),
        "logs": os.path.join(base_dir, "logs"),
        "scripts": os.path.join(base_dir, "scripts"),
        "notebooks": os.path.join(base_dir, "notebooks"),
        "evaluation": os.path.join(base_dir, "evaluation"),
    }

    logger.info("Setting up project directory structure...")

    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

    logger.info("Project directory structure setup complete")
    return directories


def check_nvidia_smi() -> Dict[str, Any]:
    """Check NVIDIA GPU information using nvidia-smi.

    Returns:
        Dictionary of GPU information
    """
    logger.info("Checking GPU information...")

    gpu_info = {
        "available": False,
        "name": None,
        "memory_total": None,
        "driver_version": None,
    }

    try:
        # Run nvidia-smi
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], text=True)

        # GPU is available
        gpu_info["available"] = True

        # Parse GPU name
        import re

        name_match = re.search(r"(\d+:[0-9A-Fa-f]+).*\| (.*) \|", nvidia_smi_output)
        if name_match:
            gpu_info["name"] = name_match.group(2).strip()

        # Parse memory
        memory_match = re.search(r"(\d+)MiB / (\d+)MiB", nvidia_smi_output)
        if memory_match:
            gpu_info["memory_used"] = int(memory_match.group(1))
            gpu_info["memory_total"] = int(memory_match.group(2))

        # Parse driver version
        driver_match = re.search(r"Driver Version: (\d+\.\d+\.\d+)", nvidia_smi_output)
        if driver_match:
            gpu_info["driver_version"] = driver_match.group(1)

        logger.info(f"GPU information: {gpu_info}")

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("NVIDIA GPU not available or nvidia-smi not found")

    return gpu_info


def check_torch_cuda() -> Dict[str, Any]:
    """Check PyTorch CUDA information.

    Returns:
        Dictionary of PyTorch CUDA information
    """
    logger.info("Checking PyTorch CUDA information...")

    cuda_info = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch

        cuda_info["torch_version"] = torch.__version__
        cuda_info["cuda_available"] = torch.cuda.is_available()

        if cuda_info["cuda_available"]:
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()

            for i in range(cuda_info["device_count"]):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory
                    // (1024 * 1024),
                }
                cuda_info["devices"].append(device_info)

        logger.info(f"PyTorch CUDA information: {cuda_info}")

    except ImportError:
        logger.warning("PyTorch not installed")

    return cuda_info


def setup_google_drive_sync(project_dir: str, drive_dir: str) -> Dict[str, str]:
    """Set up synchronization with Google Drive.

    Args:
        project_dir: Path to the project directory
        drive_dir: Path to the Google Drive directory

    Returns:
        Dictionary of sync paths
    """
    if not is_colab_runtime():
        logger.warning("Not running in Google Colab, skipping Google Drive sync setup")
        return {}

    logger.info("Setting up Google Drive synchronization...")

    # Create a directory for the project in Google Drive
    drive_project_dir = os.path.join(drive_dir, "mistral_finetuning")
    os.makedirs(drive_project_dir, exist_ok=True)

    # Create subdirectories for models and data
    drive_models_dir = os.path.join(drive_project_dir, "models")
    drive_data_dir = os.path.join(drive_project_dir, "data")
    drive_logs_dir = os.path.join(drive_project_dir, "logs")

    os.makedirs(drive_models_dir, exist_ok=True)
    os.makedirs(drive_data_dir, exist_ok=True)
    os.makedirs(drive_logs_dir, exist_ok=True)

    logger.info(f"Created Google Drive directories: {drive_project_dir}")

    sync_paths = {
        "project_dir": project_dir,
        "drive_project_dir": drive_project_dir,
        "models_dir": os.path.join(project_dir, "models"),
        "drive_models_dir": drive_models_dir,
        "data_dir": os.path.join(project_dir, "data"),
        "drive_data_dir": drive_data_dir,
        "logs_dir": os.path.join(project_dir, "logs"),
        "drive_logs_dir": drive_logs_dir,
    }

    logger.info("Google Drive synchronization setup complete")
    logger.info(f"Sync paths: {sync_paths}")

    return sync_paths


def sync_to_drive(sync_paths: Dict[str, str]) -> None:
    """Synchronize files from local storage to Google Drive.

    Args:
        sync_paths: Dictionary of sync paths
    """
    if not is_colab_runtime() or not sync_paths:
        logger.warning(
            "Not running in Google Colab or sync paths not set, skipping sync to drive"
        )
        return

    logger.info("Synchronizing files to Google Drive...")

    try:
        # Sync models
        if os.path.exists(sync_paths["models_dir"]):
            from checkpointing import save_to_google_drive

            logger.info("Syncing models...")
            save_to_google_drive(
                sync_paths["models_dir"], sync_paths["drive_models_dir"]
            )

        # Sync logs
        if os.path.exists(sync_paths["logs_dir"]):
            from checkpointing import save_to_google_drive

            logger.info("Syncing logs...")
            save_to_google_drive(sync_paths["logs_dir"], sync_paths["drive_logs_dir"])

        logger.info("Synchronization to Google Drive complete")

    except Exception as e:
        logger.error(f"Error synchronizing to Google Drive: {e}")


def sync_from_drive(sync_paths: Dict[str, str]) -> None:
    """Synchronize files from Google Drive to local storage.

    Args:
        sync_paths: Dictionary of sync paths
    """
    if not is_colab_runtime() or not sync_paths:
        logger.warning(
            "Not running in Google Colab or sync paths not set, skipping sync from drive"
        )
        return

    logger.info("Synchronizing files from Google Drive...")

    try:
        # Sync models
        if os.path.exists(sync_paths["drive_models_dir"]):
            from checkpointing import load_from_google_drive

            logger.info("Syncing models...")
            load_from_google_drive(
                sync_paths["drive_models_dir"], sync_paths["models_dir"]
            )

        # Sync data
        if os.path.exists(sync_paths["drive_data_dir"]):
            from checkpointing import load_from_google_drive

            logger.info("Syncing data...")
            load_from_google_drive(sync_paths["drive_data_dir"], sync_paths["data_dir"])

        logger.info("Synchronization from Google Drive complete")

    except Exception as e:
        logger.error(f"Error synchronizing from Google Drive: {e}")


def display_environment_info() -> None:
    """Display information about the environment."""
    if not is_colab_runtime():
        logger.warning("Not running in Google Colab, skipping environment info display")
        return

    logger.info("Displaying environment information...")

    # Check GPU
    gpu_info = check_nvidia_smi()
    cuda_info = check_torch_cuda()

    # Create HTML for display
    html = "<div style='background-color:#f9f9f9; padding:10px; border-radius:5px;'>"
    html += "<h2 style='color:#4285F4;'>Environment Information</h2>"

    # GPU Information
    html += "<h3>GPU Information</h3>"

    if gpu_info["available"]:
        html += f"<p><b>GPU:</b> {gpu_info.get('name', 'Unknown')}</p>"
        html += f"<p><b>Memory:</b> {gpu_info.get('memory_total', 'Unknown')} MiB</p>"
        html += f"<p><b>Driver:</b> {gpu_info.get('driver_version', 'Unknown')}</p>"
    else:
        html += "<p style='color:red;'><b>No GPU Available</b></p>"

    # PyTorch Information
    html += "<h3>PyTorch Information</h3>"
    html += f"<p><b>Version:</b> {cuda_info.get('torch_version', 'Not installed')}</p>"

    if cuda_info.get("cuda_available", False):
        html += f"<p><b>CUDA:</b> {cuda_info.get('cuda_version', 'Unknown')}</p>"
        html += f"<p><b>Devices:</b> {cuda_info.get('device_count', 0)}</p>"

        for device in cuda_info.get("devices", []):
            html += f"<p style='margin-left:20px;'><b>Device {device['index']}:</b> {device['name']} ({device['memory_total']} MiB)</p>"
    else:
        html += "<p style='color:orange;'><b>CUDA Not Available</b></p>"

    html += "</div>"

    # Display in Colab
    display(HTML(html))

    logger.info("Environment information displayed")


def setup_auto_save_hook(
    sync_paths: Dict[str, str], interval_minutes: int = 30
) -> None:
    """Set up a hook to automatically save model checkpoints to Google Drive.

    Args:
        sync_paths: Dictionary of sync paths
        interval_minutes: Interval in minutes to save checkpoints
    """
    if not is_colab_runtime() or not sync_paths:
        logger.warning(
            "Not running in Google Colab or sync paths not set, skipping auto-save hook setup"
        )
        return

    import threading

    def auto_save_thread():
        while True:
            # Sleep for the specified interval
            time.sleep(interval_minutes * 60)

            # Sync to drive
            logger.info(f"Auto-save hook triggered after {interval_minutes} minutes")
            sync_to_drive(sync_paths)

    # Start the thread
    auto_save_thread = threading.Thread(target=auto_save_thread, daemon=True)
    auto_save_thread.start()

    logger.info(f"Auto-save hook set up with interval of {interval_minutes} minutes")


def setup_colab_environment() -> Dict[str, Any]:
    """Set up the Google Colab environment for fine-tuning.

    Returns:
        Dictionary of environment information
    """
    logger.info("Setting up Google Colab environment...")

    if not is_colab_runtime():
        logger.warning("Not running in Google Colab, skipping environment setup")
        return {}

    # Check GPU
    gpu_info = check_nvidia_smi()
    if not gpu_info["available"]:
        logger.warning(
            "No GPU available in Google Colab. This will make fine-tuning very slow."
        )

    # Install requirements
    install_requirements()

    # Mount Google Drive
    drive_dir = mount_google_drive()

    # Set up project structure
    project_structure = setup_project_structure()

    # Set up Google Drive sync
    sync_paths = setup_google_drive_sync(project_structure["base"], drive_dir)

    # Display environment info
    display_environment_info()

    # Set up auto-save hook
    setup_auto_save_hook(sync_paths)

    # Sync from drive
    sync_from_drive(sync_paths)

    logger.info("Google Colab environment setup complete")

    return {
        "gpu_info": gpu_info,
        "torch_cuda_info": check_torch_cuda(),
        "project_structure": project_structure,
        "sync_paths": sync_paths,
    }


if __name__ == "__main__":
    if is_colab_runtime():
        setup_colab_environment()
    else:
        logger.info("This script is intended to be run in Google Colab")
        logger.info("If you're running this locally, you can ignore this message")
