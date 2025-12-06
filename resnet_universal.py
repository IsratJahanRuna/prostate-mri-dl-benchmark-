# resnet_universal.py
"""
Universal ResNet-18 trainer/evaluator for ADC, DWI, and T2 slice datasets
(extended to produce best.pt, patient-level CSVs, test summary, config, seed, splits, master metadata)

New artifacts:
- best.pt
- splits.json
- prostate158_master.csv
- val_patients.csv, test_patients.csv
- test_patient_preds.csv (with y_hat at threshold_J)
- test_eval.json
- config_used.yaml
- seed.txt
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Iterable
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve

# ----------------------
# Repro & filesystem
# ----------------------
def set_seed(seed: int = 42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ----------------------
# ID parsing (patient)
# ----------------------
def parse_patient_id(path: str) -> str:
    # e.g., "020_DWI_001.png" -> "020"
    base = os.path.basename(path); name, _ = os.path.splitext(base)
    return name.split('_')[0]

# ----------------------
# Simple YAML writer (no dependency)
# ----------------------
def dump_yaml_like(d: dict, fp: Path):
    def to_yaml_lines(k, v, indent=0) -> List[str]:
        pre = "  " * indent
        if isinstance(v, dict):
            out = [f"{pre}{k}:"]
            for kk, vv in v.items():
                out += to_yaml_lines(kk, vv, indent+1)
            return out
        else:
            if isinstance(v, str):
                v_str = f"\"{v}\""
            elif isinstance(v, bool):
                v_str = "true" if v else "false"
            else:
                v_str = str(v)
            return [f"{pre}{k}: {v_str}"]
    lines = []
    for k, v in d.items(): lines += to_yaml_lines(k, v, 0)
    fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ----------------------
# Metrics helpers
# ----------------------
def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[int,int,int,int]:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, fn, tn

def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else float('nan')
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return {'accuracy':acc,'sensitivity':sens,'specificity':spec,'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def roc_youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr; idx = int(np.argmax(j))
    return float(thr[idx]), pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thr})

# ----------------------
# Data pipeline (patient-level split)
# ----------------------
def build_imagefolder(data_dir: Path):
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return datasets.ImageFolder(str(data_dir), transform=tfm)

def stratified_patient_split(ds: datasets.ImageFolder, seed=42):
    # Map patient -> slice indices & label (derived from first slice)
    patient_to_indices: Dict[str, List[int]] = {}
    patient_label: Dict[str, int] = {}
    for i,(fp,lbl) in enumerate(ds.samples):
        pid = parse_patient_id(fp)
        patient_to_indices.setdefault(pid, []).append(i)
        if pid not in patient_label: patient_label[pid] = lbl

    patients = np.array(list(patient_to_indices.keys()))
    labels   = np.array([patient_label[p] for p in patients])

    # 20% test; from remaining 80%, 12.5% goes to val (i.e., 10% of full)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(patients, labels))
    p_trainval, p_test = patients[trainval_idx], patients[test_idx]
    y_trainval = labels[trainval_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    tr_rel, val_rel = next(sss2.split(p_trainval, y_trainval))
    p_train, p_val = p_trainval[tr_rel], p_trainval[val_rel]

    # Expand back to slice indices
    def gather(pids: Iterable[str]) -> np.ndarray:
        idxs = []
        for pid in pids: idxs.extend(patient_to_indices[pid])
        return np.array(sorted(idxs))

    idx_train = gather(p_train); idx_val = gather(p_val); idx_test = gather(p_test)

    return (Subset(ds, idx_train), Subset(ds, idx_val), Subset(ds, idx_test), 
            patient_to_indices, patient_label, list(p_train), list(p_val), list(p_test))

def make_loaders(ds_train, ds_val, ds_test, batch_size=32, workers=4):
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True),
        DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    )

# ----------------------
# Model & training
# ----------------------
def build_model(device: torch.device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, 2)
    net.to(device)
    return net

def compute_class_weights(ds: Subset | datasets.ImageFolder, num_classes=2) -> torch.Tensor:
    if isinstance(ds, Subset):
        labels = [ds.dataset.samples[i][1] for i in ds.indices]
    else:
        labels = [lbl for _, lbl in ds.samples]
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); loss_sum=0.0; total=0
    for imgs, labels in loader:
        imgs=imgs.to(device); labels=labels.to(device)
        optimizer.zero_grad()
        logits=model(imgs)
        loss=criterion(logits, labels)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*labels.size(0); total += labels.size(0)
    return loss_sum/max(1,total)

@torch.no_grad()
def collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()
    probs=[]; labels=[]; filepaths=[]; patient_ids=[]
    # forward pass
    for imgs, lbls in loader:
        imgs=imgs.to(device)
        p = torch.softmax(model(imgs), dim=1)[:,1]
        probs.append(p.cpu().numpy()); labels.append(lbls.numpy())
    # metadata
    ds = loader.dataset
    idxs = ds.indices if isinstance(ds, Subset) else range(len(ds))
    for i in idxs:
        fp = ds.dataset.samples[i][0] if isinstance(ds, Subset) else ds.samples[i][0]
        filepaths.append(fp); patient_ids.append(parse_patient_id(fp))
    return np.concatenate(probs), np.concatenate(labels), filepaths, patient_ids

def aggregate_to_patients(patient_ids: List[str], probs: np.ndarray, labels: np.ndarray):
    df = pd.DataFrame({'patient_id':patient_ids, 'p':probs, 'y_true':labels})
    # ensure one label per patient (take first; your data uses folder label consistently)
    df_p = (df.groupby('patient_id')
              .agg(p=('p','mean'), y_true=('y_true','first'))
              .reset_index())
    # sanity cast to int
    df_p['y_true'] = df_p['y_true'].astype(int)
    return df_p

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Universal ResNet-18 for ADC/DWI/T2 (patient-level outputs)")
    ap.add_argument('--dataset', required=True, choices=['ADC','DWI','T2'])
    ap.add_argument('--data_root', required=True, type=str)
    ap.add_argument('--out_dir', required=True, type=str)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    args = ap.parse_args()

    # Setup run
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_root) / args.dataset
    if not data_dir.exists(): raise FileNotFoundError(f"Data directory not found: {data_dir}")
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = Path(args.out_dir) / args.dataset / run_id
    ensure_dir(out_base)

    # Save seed
    (out_base / 'seed.txt').write_text(str(args.seed) + "\n", encoding="utf-8")

    # Data
    ds_all = build_imagefolder(data_dir)
    ds_train, ds_val, ds_test, patient_to_indices, patient_label, p_train, p_val, p_test = stratified_patient_split(ds_all, seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(ds_train, ds_val, ds_test, batch_size=args.batch_size, workers=args.workers)

    # Save splits.json
    with open(out_base/'splits.json','w') as f:
        json.dump({'train':p_train,'val':p_val,'test':p_test}, f, indent=2)

    # Master metadata (one row per slice)
    modality = args.dataset
    rows=[]
    for split_name, subset in [('train', ds_train), ('val', ds_val), ('test', ds_test)]:
        idxs = subset.indices
        for i in idxs:
            fp, lbl = ds_all.samples[i]
            rows.append({
                'patient_id': parse_patient_id(fp),
                'label': int(lbl),
                'path': fp,
                'split': split_name,
                'modality': modality
            })
    pd.DataFrame(rows).to_csv(out_base/'prostate158_master.csv', index=False)

    # Model
    model = build_model(device)
    class_weights = compute_class_weights(ds_train, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training + best checkpoint (patient-level val AUC)
    best_auc = -1.0
    best_path = out_base / 'best.pt'
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # validate (slice-level forward, aggregate to patients, AUC on patients)
        val_prob, val_y, val_fp, val_pid = collect_probs(model, val_loader, device)
        df_val_pat = aggregate_to_patients(val_pid, val_prob, val_y)
        val_auc = roc_auc_score(df_val_pat['y_true'].values, df_val_pat['p'].values) if len(df_val_pat['y_true'].unique())==2 else float('nan')
        print(f"Epoch {epoch}/{args.epochs} - train_loss={tr_loss:.4f} - val_auc_pat={val_auc:.4f}")
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)

    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device))

    # ---- Final VAL (patients) + threshold_J on patients ----
    val_prob, val_y, val_fp, val_pid = collect_probs(model, val_loader, device)
    df_val_pat = aggregate_to_patients(val_pid, val_prob, val_y)
    df_val_pat[['patient_id','y_true','p']].to_csv(out_base/'val_patients.csv', index=False)

    thr_J, roc_val_df = roc_youden_j_threshold(df_val_pat['y_true'].values, df_val_pat['p'].values)
    roc_val_df.to_csv(out_base/'val_roc_patients.csv', index=False)  # keep a patient-level ROC too

    # ---- TEST (patients) ----
    test_prob, test_y, test_fp, test_pid = collect_probs(model, test_loader, device)
    df_test_pat = aggregate_to_patients(test_pid, test_prob, test_y)
    df_test_pat[['patient_id','y_true','p']].to_csv(out_base/'test_patients.csv', index=False)

    # Binary decisions at threshold_J (from VAL patients)
    y_hat = (df_test_pat['p'].values >= thr_J).astype(int)
    df_preds = df_test_pat.copy()
    df_preds['y_hat'] = y_hat
    df_preds[['patient_id','y_true','p','y_hat']].to_csv(out_base/'test_patient_preds.csv', index=False)

    # Metrics at patient-level on TEST
    test_auc = roc_auc_score(df_test_pat['y_true'].values, df_test_pat['p'].values) if len(df_test_pat['y_true'].unique())==2 else float('nan')
    tp, fp, fn, tn = confusion_at_threshold(df_test_pat['y_true'].values, df_test_pat['p'].values, thr_J)
    m = metrics_from_counts(tp, fp, fn, tn)

    with open(out_base/'test_eval.json','w') as f:
        json.dump({
            'threshold_J': float(thr_J),
            'test_auc': float(test_auc),
            'test_acc': float(m['accuracy']),
            'test_sens': float(m['sensitivity']),
            'test_spec': float(m['specificity']),
        }, f, indent=2)

    # ---- Config YAML ----
    cfg = {
        'run_name': f"resnet_{args.dataset}_seed{args.seed}",
        'seed': args.seed,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_root': str(Path(args.data_root).resolve()),
        'split_csv': str((out_base/'prostate158_master.csv').resolve()),
        'img_size': '224x224',
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'model': {'name': 'resnet18', 'pretrained': True},
        'optim': {'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': [0.9, 0.999]},
        'train': {'epochs': args.epochs, 'early_stop_patience': 0},  # no ES; we track best by val AUC
        'logging': {'out_dir': str(out_base.resolve())}
    }
    dump_yaml_like(cfg, out_base/'config_used.yaml')

    print(f"Run complete. Outputs in: {out_base}")
    print("Saved: best.pt, splits.json, prostate158_master.csv, val_patients.csv, test_patients.csv, "
          "test_patient_preds.csv, test_eval.json, config_used.yaml, seed.txt")

if __name__ == '__main__':
    main()
