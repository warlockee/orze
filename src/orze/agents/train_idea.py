#!/usr/bin/env python3
"""
Example training script for orze.

Trains a CIFAR-10 classifier based on config from ideas.md.
Demonstrates the contract that the orchestrator expects:
  - Input: CUDA_VISIBLE_DEVICES env, --idea-id, --results-dir, --ideas-md, --config
  - Output: results/{idea_id}/metrics.json with {"status": "COMPLETED"|"FAILED", ...}
"""

import argparse
import json
import os
import re
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml


def _atomic_write_json(path: Path, data: dict):
    """Write JSON atomically via tmp+replace to prevent partial reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_host = "".join(c if c.isalnum() else "_" for c in socket.gethostname())
    tmp = path.with_name(f"{path.name}.{safe_host}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Simple 3-layer CNN."""
    def __init__(self, channels=(32, 64, 128), num_classes=10):
        super().__init__()
        layers = []
        in_c = 3
        for c in channels:
            layers += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c),
                       nn.ReLU(), nn.MaxPool2d(2)]
            in_c = c
        self.features = nn.Sequential(*layers)
        # After 3 pools on 32x32: 4x4
        feat_size = channels[-1] * (32 // (2 ** len(channels))) ** 2
        self.classifier = nn.Linear(feat_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNetSmall(nn.Module):
    """Small ResNet with skip connections."""
    def __init__(self, channels=(64, 128), blocks_per_stage=2, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(),
        )
        stages = []
        in_c = channels[0]
        for c in channels:
            if c != in_c:
                stages.append(nn.Sequential(
                    nn.Conv2d(in_c, c, 1), nn.BatchNorm2d(c), nn.ReLU(),
                ))
                in_c = c
            for _ in range(blocks_per_stage):
                stages.append(ResBlock(c))
            stages.append(nn.MaxPool2d(2))
        self.stages = nn.Sequential(*stages)
        feat_size = channels[-1] * (32 // (2 ** len(channels))) ** 2
        self.classifier = nn.Linear(feat_size, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class WideCNN(nn.Module):
    """Wide CNN with fewer layers but more channels."""
    def __init__(self, width=256, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(model_cfg: dict) -> nn.Module:
    model_type = model_cfg.get("type", "simple_cnn")
    if model_type == "simple_cnn":
        channels = model_cfg.get("channels", [32, 64, 128])
        return SimpleCNN(channels=channels)
    elif model_type == "resnet_small":
        channels = model_cfg.get("channels", [64, 128])
        blocks = model_cfg.get("blocks_per_stage", 2)
        return ResNetSmall(channels=channels, blocks_per_stage=blocks)
    elif model_type == "wide_cnn":
        width = model_cfg.get("width", 256)
        return WideCNN(width=width)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Ideas parsing (same logic as the orchestrator)
# ---------------------------------------------------------------------------

def get_idea_config(ideas_md: str, idea_id: str) -> dict:
    """Extract a single idea's YAML config from ideas.md."""
    try:
        text = Path(ideas_md).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    pattern = re.compile(r"^## (idea-[a-z0-9]+):\s*(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        if m.group(1) != idea_id:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[start:end]
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", raw, re.DOTALL)
        if yaml_match:
            return yaml.safe_load(yaml_match.group(1)) or {}
    return {}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    start_time = time.time()
    results_dir = Path(args.results_dir) / args.idea_id

    # Load configs: base config merged with idea-specific overrides
    base_cfg = {}
    if Path(args.config).exists():
        base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    idea_cfg = get_idea_config(args.ideas_md, args.idea_id)

    # Merge: idea config overrides base
    cfg = {**base_cfg}
    for key in idea_cfg:
        if isinstance(idea_cfg[key], dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key] = {**cfg[key], **idea_cfg[key]}
        else:
            cfg[key] = idea_cfg[key]

    train_cfg = cfg.get("training") or {}
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    ckpt_cfg = cfg.get("checkpointing") or {}

    epochs = train_cfg.get("epochs", 10)
    batch_size = train_cfg.get("batch_size", 128)
    lr = train_cfg.get("lr", 0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    print(f"[{args.idea_id}] Device: {device}")
    print(f"[{args.idea_id}] Config: {json.dumps(cfg, indent=2)}")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_dir = Path(data_cfg.get("data_dir") or "./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    num_workers = data_cfg.get("num_workers", 2)

    # Cross-process lock for dataset download (prevents corruption when
    # multiple GPUs launch simultaneously on a fresh machine)
    lock_dir = data_dir / ".download_lock"

    marker = data_dir / ".download_complete"
    if not marker.exists():
        acquired_lock = False
        while True:
            try:
                lock_dir.mkdir(exist_ok=False)
                acquired_lock = True
                break
            except FileExistsError:
                if marker.exists():
                    break
                # Break stale locks (e.g. downloader was SIGKILLed)
                try:
                    if time.time() - lock_dir.stat().st_mtime > 300:
                        lock_dir.rmdir()
                except OSError:
                    pass
                time.sleep(2)

        if acquired_lock:
            try:
                if not marker.exists():
                    torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform_train)
                    torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform_test)
                    marker.touch()
            finally:
                if lock_dir.exists():
                    try:
                        lock_dir.rmdir()
                    except Exception:
                        pass

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = build_model(model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.idea_id}] Model: {model_cfg.get('type', 'simple_cnn')} ({num_params:,} params)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= total
        test_acc = correct / total

        print(f"[{args.idea_id}] Epoch {epoch+1}/{epochs} — "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        })

        # Checkpoint
        if ckpt_cfg.get("save_best", True) and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), results_dir / "best_model.pt")

    training_time = time.time() - start_time

    # Write final metrics
    metrics = {
        "status": "COMPLETED",
        "idea_id": args.idea_id,
        "model_type": model_cfg.get("type", "simple_cnn"),
        "num_params": num_params,
        "test_accuracy": best_acc,
        "test_loss": history[-1]["test_loss"] if history else 0,
        "training_time": training_time,
        "epochs": epochs,
        "history": history,
        "config": cfg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_write_json(results_dir / "metrics.json", metrics)
    print(f"[{args.idea_id}] Done! best_acc={best_acc:.4f} in {training_time:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 model from an idea config")
    parser.add_argument("--idea-id", required=True, help="Idea ID (e.g. idea-001)")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--ideas-md", required=True, help="Path to ideas.md")
    parser.add_argument("--config", required=True, help="Path to base config YAML")
    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        # Write failure metrics so the orchestrator knows what happened
        results_dir = Path(args.results_dir) / args.idea_id
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "status": "FAILED",
            "error": str(e),
            "idea_id": args.idea_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _atomic_write_json(results_dir / "metrics.json", metrics)
        print(f"[{args.idea_id}] FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
