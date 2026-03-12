#!/usr/bin/env python3
"""
Download VILA-R aesthetic scorer weights from Google Cloud Storage.

VILA-R is the aesthetic scorer used in the Cropper paper.
Weights are available at: https://console.cloud.google.com/storage/browser/gresearch/vila
"""

import os
import subprocess
import sys
from pathlib import Path


def download_vila_weights():
    """Download VILA-R weights from Google Cloud Storage."""

    # Target directory
    weights_dir = Path(__file__).parent.parent / "weights"
    vila_dir = weights_dir / "vila_rank_tuned"

    weights_dir.mkdir(parents=True, exist_ok=True)

    if vila_dir.exists():
        print(f"VILA-R weights already exist at {vila_dir}")
        return True

    print("Downloading VILA-R weights from Google Cloud Storage...")
    print("This may take a few minutes...")

    # Google Cloud Storage URL for VILA-R
    gcs_url = "gs://gresearch/vila/vila_rank_tuned"

    # Try using gsutil
    try:
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", gcs_url, str(weights_dir)],
            check=True,
        )
        print(f"VILA-R weights downloaded to {vila_dir}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"gsutil failed: {e}")

    # Alternative: try using gcloud storage
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", gcs_url, str(weights_dir)],
            check=True,
        )
        print(f"VILA-R weights downloaded to {vila_dir}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"gcloud storage failed: {e}")

    # If both fail, provide manual instructions
    print("\n" + "="*60)
    print("AUTOMATIC DOWNLOAD FAILED")
    print("="*60)
    print("\nPlease download VILA-R weights manually:")
    print("\n1. Install Google Cloud SDK:")
    print("   pip install google-cloud-storage")
    print("   # or: https://cloud.google.com/sdk/docs/install")
    print("\n2. Download weights:")
    print(f"   gsutil -m cp -r {gcs_url} {weights_dir}")
    print("\n3. Or download from browser:")
    print("   https://console.cloud.google.com/storage/browser/gresearch/vila")
    print(f"   Save to: {vila_dir}")
    print("="*60)

    return False


def verify_vila_weights():
    """Verify VILA-R weights are correctly downloaded."""
    weights_dir = Path(__file__).parent.parent / "weights"
    vila_dir = weights_dir / "vila_rank_tuned"

    # Check for SavedModel format
    saved_model_pb = vila_dir / "saved_model.pb"

    if saved_model_pb.exists():
        print(f"VILA-R weights verified at {vila_dir}")
        return True

    # Check for checkpoint format
    checkpoint_files = list(vila_dir.glob("*.ckpt*")) + list(vila_dir.glob("checkpoint"))
    if checkpoint_files:
        print(f"VILA-R checkpoint found at {vila_dir}")
        return True

    print(f"VILA-R weights not found or incomplete at {vila_dir}")
    return False


def test_vila_loading():
    """Test loading VILA-R model."""
    try:
        import tensorflow as tf

        weights_dir = Path(__file__).parent.parent / "weights"
        vila_dir = weights_dir / "vila_rank_tuned"

        if not vila_dir.exists():
            print("VILA-R weights not found. Run download first.")
            return False

        print("Loading VILA-R model...")
        model = tf.saved_model.load(str(vila_dir))
        print("VILA-R model loaded successfully!")

        # Test inference
        import numpy as np
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model(test_image)
        print(f"Test output shape: {output.shape if hasattr(output, 'shape') else type(output)}")

        return True

    except ImportError:
        print("TensorFlow not installed. Run: pip install tensorflow")
        return False
    except Exception as e:
        print(f"Error loading VILA-R: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VILA-R weights")
    parser.add_argument("--verify", action="store_true", help="Verify existing weights")
    parser.add_argument("--test", action="store_true", help="Test loading the model")

    args = parser.parse_args()

    if args.verify:
        verify_vila_weights()
    elif args.test:
        test_vila_loading()
    else:
        success = download_vila_weights()
        if success:
            verify_vila_weights()
