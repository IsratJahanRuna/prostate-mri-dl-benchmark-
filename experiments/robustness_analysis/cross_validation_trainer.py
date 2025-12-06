import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from collections import defaultdict


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_patient_id(path: str) -> str:
    """
    Extract patient_id from a filename. Assumes filenames like:
      <patient>_... .png/jpg
    e.g., 'P123_slice_001.png' -> 'P123'
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name.split('_')[0]


# ---------------------------
# Data / Model Builders
# ---------------------------
def build_imagefolder(data_dir: Path) -> datasets.ImageFolder:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = datasets.ImageFolder(str(data_dir), transform=tfm)
        if len(dataset) == 0:
            raise RuntimeError(f"No valid images found in {data_dir}")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Error loading dataset from {data_dir}: {str(e)}")


def build_model(model_name: str, device: torch.device) -> nn.Module:
    try:
        if model_name == 'resnet':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 2)
        elif model_name == 'densenet':
            net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_features = net.classifier.in_features
            net.classifier = nn.Linear(in_features, 2)
        elif model_name == 'efficientnet':
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_features, 2)
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

        net.to(device)
        return net
    except Exception as e:
        raise RuntimeError(f"Error building {model_name} model: {str(e)}")


# ---------------------------
# Metrics & Training
# ---------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, sensitivity, specificity, AUC."""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'auc': float(auc),
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return float(total_loss / max(total_samples, 1))


# ---------------------------
# CV Splitter (Patient-level)
# ---------------------------
def prepare_patient_cv_splits(dataset: datasets.ImageFolder, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Patient-level CV:
      - Group all image indices by patient_id
      - Stratify by patient-level label using first index's label per patient
    """
    patient_to_indices = defaultdict(list)
    patient_labels = {}

    for idx, (path, label) in enumerate(dataset.samples):
        patient_id = parse_patient_id(path)
        patient_to_indices[patient_id].append(idx)
        # Assume a single label per patient (consistent across patient images)
        if patient_id not in patient_labels:
            patient_labels[patient_id] = label

    patients = np.array(list(patient_to_indices.keys()))
    labels = np.array([patient_labels[p] for p in patients])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_splits = []

    for train_idx, val_idx in skf.split(patients, labels):
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]

        train_indices, val_indices = [], []
        for p in train_patients:
            train_indices.extend(patient_to_indices[p])
        for p in val_patients:
            val_indices.extend(patient_to_indices[p])

        cv_splits.append((np.array(train_indices), np.array(val_indices)))

    return cv_splits


# ---------------------------
# One Seed (possibly many folds)
# ---------------------------
def run_cv_experiment(args, seed: int, out_dir: Path) -> Dict[str, List[float]]:
    """
    Run one repeated-CV with a specific seed. Saves per-fold artifacts:
      seed_<seed>/fold_<k>/{splits.json, best.pt, performance_metrics.json}
    Returns dict of lists with per-fold metrics for this seed.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset for this run
    data_dir = Path(args.data_root) / args.dataset
    dataset = build_imagefolder(data_dir)

    # Prepare patient-level CV splits (K folds)
    cv_splits = prepare_patient_cv_splits(dataset, args.n_splits, seed)

    # Seed-level output directory
    seed_dir = out_dir / f"seed_{seed}"
    ensure_dir(seed_dir)

    # Storage for fold metrics
    fold_metrics = defaultdict(list)

    # K-fold loop
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        print(f"\nFold {fold}/{args.n_splits}  (seed={seed})")

        # Per-fold dir + save splits.json (patient_ids + indices)
        fold_dir = seed_dir / f"fold_{fold}"
        ensure_dir(fold_dir)

        # Save splits.json
        train_pids = sorted({parse_patient_id(dataset.samples[i][0]) for i in train_idx})
        val_pids = sorted({parse_patient_id(dataset.samples[i][0]) for i in val_idx})
        with open(fold_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump({
                "train_patient_ids": train_pids,
                "val_patient_ids": val_pids,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            }, f, indent=2)

        # Build data loaders
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Model + training bits
        model = build_model(args.model, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        best_val_auc = -1.0
        best_epoch = 0
        patience_counter = 0

        # Train with early stopping, save best.pt on improvement
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, device)

            print(f"Epoch {epoch}/{args.epochs} - "
                  f"Loss: {train_loss:.4f} - "
                  f"Val AUC: {val_metrics['auc']:.4f}")

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                # Save best weights
                torch.save(model.state_dict(), fold_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (no AUC improvement).")
                break

        # Reload best.pt before final evaluation
        if (fold_dir / "best.pt").exists():
            model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))

        final_metrics = evaluate_model(model, val_loader, device)

        # Save performance_metrics.json (includes best epoch info)
        with open(fold_dir / "performance_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "final_metrics": final_metrics,
                "best_val_auc": float(best_val_auc),
                "best_epoch": int(best_epoch)
            }, f, indent=2)

        # Accumulate for this seed
        for metric, value in final_metrics.items():
            fold_metrics[metric].append(value)

    return dict(fold_metrics)


# ---------------------------
# Orchestrator
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-validation Trainer for Prostate Cancer Detection")
    parser.add_argument('--model', required=True, choices=['resnet', 'densenet', 'efficientnet'],
                        help="Model architecture to use")
    parser.add_argument('--dataset', required=True, choices=['ADC', 'DWI', 'T2'],
                        help="Dataset/modality to use")
    parser.add_argument('--data_root', required=True, type=str,
                        help="Root directory containing the dataset folders")
    parser.add_argument('--out_dir', required=True, type=str,
                        help="Directory to save outputs")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Training batch size")
    parser.add_argument('--n_splits', type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument('--n_repeats', type=int, default=5,
                        help="Number of times to repeat CV with different seeds")
    parser.add_argument('--base_seed', type=int, default=42,
                        help="Base random seed (repeats use base_seed + i)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument('--patience', type=int, default=2,
                        help="Early stopping patience (epochs without AUC improvement)")
    args = parser.parse_args()

    # Output root: <out_dir>/<model>/<dataset>/cv_<timestamp>/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) / args.model / args.dataset / f"cv_{timestamp}"
    ensure_dir(out_dir)

    # Save full configuration
    config = vars(args).copy()
    config["resolved_out_dir"] = str(out_dir)
    with open(out_dir / 'config.json', 'w', encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Repeats: run CV with different seeds
    all_results = []
    metrics = ['accuracy', 'sensitivity', 'specificity', 'auc']

    for i in range(args.n_repeats):
        seed = args.base_seed + i
        print(f"\nRunning CV experiment {i + 1}/{args.n_repeats} with seed {seed}")
        results = run_cv_experiment(args, seed, out_dir)  # per-seed folders & files saved inside
        all_results.append(results)

        # Save per-seed aggregate (list per metric across folds)
        with open(out_dir / f'results_seed_{seed}.json', 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Aggregate across all folds and seeds -> final mean ± std
    final_stats = {}
    for metric in metrics:
        values = []
        for result in all_results:
            values.extend(result.get(metric, []))

        if len(values) == 0:
            mean, std = 0.0, 0.0
        else:
            mean = float(np.mean(values))
            std = float(np.std(values))

        final_stats[metric] = {
            'mean': mean,
            'std': std,
            'formatted': f"{mean:.3f} ± {std:.3f}"
        }

    with open(out_dir / 'final_statistics.json', 'w', encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)

    # Pretty print
    print("\nFinal Results:")
    print("-" * 50)
    for metric, stats in final_stats.items():
        print(f"{metric}: {stats['formatted']}")

    print(f"\nAll results saved in: {out_dir}")


if __name__ == '__main__':
    main()
