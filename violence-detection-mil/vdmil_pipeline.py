#!/usr/bin/env python3
"""
VD-MIL Complete Training Pipeline
Handles nested folders, normalization, clip generation, and training.

Usage:
    python vdmil_pipeline.py --dataset scvd --epochs 20
    python vdmil_pipeline.py --dataset both --epochs 20 --batch-size 8
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict


class VDMILPipeline:
    """Complete pipeline for training VD-MIL violence detection model."""

    def __init__(self, dataset_name, epochs=20, batch_size=8):
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size

        # Paths
        self.project_dir = Path("/home/moselhy/Desktop/Depi_Ai_Final_Project")
        self.vdmil_dir = self.project_dir / "violence-detection-mil"
        self.data_dir = self.project_dir / "in-car-violence-detection" / "data" / "raw"
        self.tmp_dir = Path(f"/tmp/vdmil_{dataset_name}")
        self.labels_dir = self.tmp_dir / "labels"
        self.fps8_dir = self.tmp_dir / "videos_8fps"
        self.clips_dir = self.tmp_dir / "clips"
        self.models_dir = self.vdmil_dir / "models" / "checkpoints" / "violence"

        # Create directories
        for d in [self.labels_dir, self.fps8_dir, self.clips_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"VD-MIL Pipeline initialized")
        print(f"  Dataset: {dataset_name}")
        print(f"  Working dir: {self.tmp_dir}")

    def run_command(self, cmd, description=""):
        """Run shell command and print output."""
        if description:
            print(f"\n{'='*50}")
            print(f"{description}")
            print(f"{'='*50}")

        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"⚠️  Command failed with code {result.returncode}")
            return False

        return True

    def step1_create_labels(self):
        """Create labels.txt files for datasets."""
        print("\n📋 STEP 1: Creating labels.txt files...")

        if self.dataset_name in ["scvd", "both"]:
            self._create_scvd_labels()

        if self.dataset_name in ["violence_in_car", "both"]:
            self._create_vic_labels()

        print("✅ Labels created!")

    def _create_scvd_labels(self):
        """Create labels for SCVD dataset."""
        scvd_base = self.data_dir / "SCVD" / "SCVD_converted"

        if not scvd_base.exists():
            print(f"⚠️  SCVD not found at {scvd_base}")
            return

        # Process each split and class
        splits = ["Train", "Test"]
        classes = {"Violence": True, "Normal": False}

        all_train_lines = []
        all_test_lines = []

        for split in splits:
            for class_name, is_violent in classes.items():
                class_dir = scvd_base / split / class_name
                if not class_dir.exists():
                    continue

                # Find all videos recursively
                videos = []
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    videos.extend(class_dir.rglob(f"*{ext}"))

                label = "0,inf" if is_violent else "-1,-1"
                lines = [f"{v.relative_to(scvd_base)} {label}\n" for v in sorted(videos)]

                if split == "Train":
                    all_train_lines.extend(lines)
                else:
                    all_test_lines.extend(lines)

                print(f"  {split}/{class_name}: {len(lines)} videos ({'violent' if is_violent else 'normal'})")

        # Save combined labels
        train_file = self.labels_dir / "scvd_train_labels.txt"
        test_file = self.labels_dir / "scvd_test_labels.txt"

        with open(train_file, 'w') as f:
            f.writelines(all_train_lines)

        with open(test_file, 'w') as f:
            f.writelines(all_test_lines)

        print(f"  Saved: {train_file} ({len(all_train_lines)} entries)")
        print(f"  Saved: {test_file} ({len(all_test_lines)} entries)")

    def _create_vic_labels(self):
        """Create labels for Violence-in-Car dataset."""
        vic_base = self.data_dir / "violence-in-car"

        if not vic_base.exists():
            print(f"⚠️  Violence-in-Car not found at {vic_base}")
            return

        classes = {"attack": True, "non_attack": False}
        all_lines = []

        for class_name, is_violent in classes.items():
            class_dir = vic_base / class_name
            if not class_dir.exists():
                continue

            videos = []
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                videos.extend(class_dir.rglob(f"*{ext}"))

            label = "0,inf" if is_violent else "-1,-1"
            lines = [f"{v.relative_to(vic_base)} {label}\n" for v in sorted(videos)]
            all_lines.extend(lines)

            print(f"  {class_name}: {len(lines)} videos ({'violent' if is_violent else 'normal'})")

        labels_file = self.labels_dir / "vic_labels.txt"
        with open(labels_file, 'w') as f:
            f.writelines(all_lines)

        print(f"  Saved: {labels_file} ({len(all_lines)} entries)")

    def step2_normalize_videos(self):
        """Normalize videos to 8fps using ffmpeg."""
        print("\n📹 STEP 2: Normalizing videos to 8fps...")

        if self.dataset_name in ["scvd", "both"]:
            self._normalize_dataset(
                self.data_dir / "SCVD" / "SCVD_converted",
                self.fps8_dir / "scvd"
            )

        if self.dataset_name in ["violence_in_car", "both"]:
            self._normalize_dataset(
                self.data_dir / "violence-in-car",
                self.fps8_dir / "violence_in_car"
            )

        print("✅ Normalization complete!")

    def _normalize_dataset(self, input_dir, output_dir):
        """Normalize all videos in input_dir to 8fps, preserving structure."""
        if not input_dir.exists():
            print(f"⚠️  Input directory not found: {input_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all videos
        videos = []
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            videos.extend(input_dir.rglob(f"*{ext}"))

        print(f"  Found {len(videos)} videos in {input_dir.name}")

        for i, video in enumerate(sorted(videos), 1):
            rel_path = video.relative_to(input_dir)
            output_file = output_dir / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            cmd = f'ffmpeg -i "{video}" -r 8 -y "{output_file}" -loglevel error'

            print(f"  [{i}/{len(videos)}] {rel_path}...", end=" ")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅")
            else:
                print(f"⚠️  (ffmpeg error)")

    def step3_flatten_and_generate_clips(self):
        """Flatten normalized videos and generate clips."""
        print("\n✂️ STEP 3: Generating 5-second clips...")

        if self.dataset_name in ["scvd", "both"]:
            self._generate_scvd_clips()

        if self.dataset_name in ["violence_in_car", "both"]:
            self._generate_vic_clips()

        print("✅ Clip generation complete!")

    def _generate_scvd_clips(self):
        """Generate clips for SCVD dataset."""
        normalized_dir = self.fps8_dir / "scvd"
        clips_output = self.clips_dir / "scvd"

        if not normalized_dir.exists():
            print(f"⚠️  Normalized SCVD not found at {normalized_dir}")
            return

        # Create flat directory with renamed videos
        flat_dir = self.fps8_dir / "scvd_flat"
        flat_dir.mkdir(parents=True, exist_ok=True)

        # Copy videos with flattened names
        video_map = {}  # flat_name -> original_relative_path

        videos = list(normalized_dir.rglob("*.mp4")) + list(normalized_dir.rglob("*.avi"))

        for video in sorted(videos):
            rel_path = video.relative_to(normalized_dir)
            flat_name = str(rel_path).replace("/", "_").replace("\\", "_")
            shutil.copy2(video, flat_dir / flat_name)
            video_map[flat_name] = str(rel_path)

        print(f"  Flattened {len(videos)} videos to {flat_dir}")

        # Create flat labels
        original_labels = self.labels_dir / "scvd_train_labels.txt"
        flat_labels = self.labels_dir / "scvd_train_labels_flat.txt"

        if original_labels.exists():
            with open(original_labels, 'r') as f:
                lines = f.readlines()

            with open(flat_labels, 'w') as f:
                for line in lines:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        orig_path = parts[0]
                        flat_name = orig_path.replace("/", "_").replace("\\", "_")
                        f.write(f"{flat_name} {parts[1]}\n")

            print(f"  Created flat labels: {flat_labels}")

        # Run generate_clips_dataset.py
        vdmil_dir = self.vdmil_dir
        cmd = f"""cd {vdmil_dir} && python3 generate_clips_dataset.py \
            --path_videos {flat_dir} \
            --path_annotations {flat_labels} \
            --path_output_clips {clips_output} \
            --length_clip 5 \
            --stride_window_clip 5 \
            --threshold 0.3"""

        self.run_command(cmd, "Generating SCVD clips")

    def _generate_vic_clips(self):
        """Generate clips for Violence-in-Car dataset."""
        normalized_dir = self.fps8_dir / "violence_in_car"
        clips_output = self.clips_dir / "violence_in_car"
        labels_file = self.labels_dir / "vic_labels.txt"

        if not normalized_dir.exists():
            print(f"⚠️  Normalized VIC not found at {normalized_dir}")
            return

        # Similar flattening process
        flat_dir = self.fps8_dir / "vic_flat"
        flat_dir.mkdir(parents=True, exist_ok=True)

        videos = list(normalized_dir.rglob("*.mp4")) + list(normalized_dir.rglob("*.avi"))

        for video in sorted(videos):
            rel_path = video.relative_to(normalized_dir)
            flat_name = str(rel_path).replace("/", "_").replace("\\", "_")
            shutil.copy2(video, flat_dir / flat_name)

        # Create flat labels
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                lines = f.readlines()

            flat_labels = self.labels_dir / "vic_labels_flat.txt"
            with open(flat_labels, 'w') as f:
                for line in lines:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        orig_path = parts[0]
                        flat_name = orig_path.replace("/", "_").replace("\\", "_")
                        f.write(f"{flat_name} {parts[1]}\n")

        cmd = f"""cd {self.vdmil_dir} && python3 generate_clips_dataset.py \
            --path_videos {flat_dir} \
            --path_annotations {flat_labels} \
            --path_output_clips {clips_output} \
            --length_clip 5 \
            --stride_window_clip 5 \
            --threshold 0.3"""

        self.run_command(cmd, "Generating Violence-in-Car clips")

    def step4_train_classifier(self):
        """Train the MIL classifier."""
        print("\n🎓 STEP 4: Training classifier...")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Expected time: ~30-60 minutes on GTX 1650")

        # Combine positive and negative clips from all datasets
        positive_dirs = []
        negative_dirs = []

        for dataset in ["scvd", "violence_in_car"]:
            pos_dir = self.clips_dir / dataset / "positive"
            neg_dir = self.clips_dir / dataset / "negative"

            if pos_dir.exists():
                positive_dirs.append(str(pos_dir))
            if neg_dir.exists():
                negative_dirs.append(str(neg_dir))

        if not positive_dirs or not negative_dirs:
            print("⚠️  No clips found! Check previous steps.")
            return False

        # Create combined directories
        combined_pos = self.clips_dir / "combined_positive"
        combined_neg = self.clips_dir / "combined_negative"

        combined_pos.mkdir(exist_ok=True)
        combined_neg.mkdir(exist_ok=True)

        # Symlink or copy clips
        for pos_dir in positive_dirs:
            for clip in Path(pos_dir).glob("*.mp4"):
                shutil.copy2(clip, combined_pos / clip.name)

        for neg_dir in negative_dirs:
            for clip in Path(neg_dir).glob("*.mp4"):
                shutil.copy2(clip, combined_neg / clip.name)

        pos_count = len(list(combined_pos.glob("*.mp4")))
        neg_count = len(list(combined_neg.glob("*.mp4")))

        print(f"  Training with {pos_count} positive + {neg_count} negative clips")

        # Run training
        cmd = f"""cd {self.vdmil_dir} && python3 train_activity_detector.py \
            --path_positive_clips {combined_pos} \
            --path_negative_clips {combined_neg} \
            --folder_backbone_model {self.vdmil_dir}/movinet_weights \
            --batch_size {self.batch_size} \
            --epochs {self.epochs} \
            --checkpoint_interval 1 \
            --length_window 8 \
            --stride 4 \
            --folder_trained_models {self.models_dir} \
            --device cuda"""

        return self.run_command(cmd, "Training classifier")

    def run(self):
        """Run complete pipeline."""
        print("="*60)
        print("VD-MIL Violence Detection - Complete Training Pipeline")
        print("="*60)

        self.step1_create_labels()
        self.step2_normalize_videos()
        self.step3_flatten_and_generate_clips()
        success = self.step4_train_classifier()

        if success:
            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETE!")
            print("="*60)
            print(f"Model saved to: {self.models_dir}/model_{self.epochs}.pt")
            print("\nNext steps:")
            print("  1. Evaluate: python evaluate_violence_vdmil.py --model ...")
            print("  2. Test inference: python real_time_detection_vdmil.py ...")
            print("  3. Git commit and push")
        else:
            print("\n⚠️  Pipeline completed with errors. Check logs above.")

        return success


def main():
    parser = argparse.ArgumentParser(description='VD-MIL Complete Training Pipeline')
    parser.add_argument('--dataset', choices=['scvd', 'violence_in_car', 'both'],
                       default='scvd', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--skip-normalization', action='store_true',
                       help='Skip normalization (if already done)')
    parser.add_argument('--skip-labels', action='store_true',
                       help='Skip label creation (if already done)')
    parser.add_argument('--skip-clips', action='store_true',
                       help='Skip clip generation (if already done)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training step')

    args = parser.parse_args()

    pipeline = VDMILPipeline(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    if args.train_only:
        pipeline.step4_train_classifier()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
