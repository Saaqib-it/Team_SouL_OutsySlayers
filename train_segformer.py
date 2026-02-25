import os
import math
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    get_linear_schedule_with_warmup,
)


# -----------------------------
# Configuration
# -----------------------------

DATA_ROOT = "/content/drive/MyDrive/hack"

TRAIN_IMAGE_DIR = os.path.join(
    DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "train", "Color_Images"
)
TRAIN_MASK_DIR  = os.path.join(
    DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "train", "Segmentation"
)

VAL_IMAGE_DIR   = os.path.join(
    DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "val", "Color_Images"
)
VAL_MASK_DIR    = os.path.join(
    DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "val", "Segmentation"
)

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

NUM_CLASSES = 10
IGNORE_INDEX = 255

NUM_EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 6e-5
WARMUP_RATIO = 0.1

BEST_MODEL_PATH = os.path.join(DATA_ROOT, "best_segformer.pth")
LAST_MODEL_PATH = os.path.join(DATA_ROOT, "last_segformer.pth")

IMAGE_SIZE = 512


# Raw mask value to contiguous class id mapping 0-9
RAW_TO_CLASS_ID: Dict[int, int] = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9,  # Sky
}

ID2LABEL = {
    0: "Trees",
    1: "Lush_Bushes",
    2: "Dry_Grass",
    3: "Dry_Bushes",
    4: "Ground_Clutter",
    5: "Flowers",
    6: "Logs",
    7: "Rocks",
    8: "Landscape",
    9: "Sky",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# -----------------------------
# Utility functions
# -----------------------------

def maybe_mount_google_drive():
    """
    If running in Google Colab, try to mount Google Drive at /content/drive.
    This will be a no-op outside Colab.
    """
    try:
        import google.colab  # type: ignore

        from google.colab import drive  # type: ignore

        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    except Exception:
        # Not in Colab or mount already handled; ignore.
        pass


def list_sorted_files(image_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    image_files = sorted(
        [f for f in os.listdir(image_dir) if not f.startswith(".")]
    )
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if not f.startswith(".")]
    )

    # Ensure matching filenames between images and masks
    image_paths: List[str] = []
    mask_paths: List[str] = []
    mask_set = set(mask_files)

    for img_name in image_files:
        if img_name in mask_set:
            image_paths.append(os.path.join(image_dir, img_name))
            mask_paths.append(os.path.join(mask_dir, img_name))

    return image_paths, mask_paths


def remap_mask_values(mask: np.ndarray) -> np.ndarray:
    """
    Convert raw mask values to contiguous class IDs 0..NUM_CLASSES-1.
    Unknown values become IGNORE_INDEX.
    """
    remapped = np.full_like(mask, IGNORE_INDEX, dtype=np.uint8)
    for raw_val, class_id in RAW_TO_CLASS_ID.items():
        remapped[mask == raw_val] = class_id
    return remapped


def compute_class_weights(
    mask_paths: List[str], num_classes: int, ignore_index: int
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from masks.
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    for m_path in mask_paths:
        mask = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        remapped = remap_mask_values(mask)
        for c in range(num_classes):
            counts[c] += np.sum(remapped == c)

    counts = np.maximum(counts, 1)  # avoid division by zero
    inv_freq = 1.0 / counts.astype(np.float64)
    inv_freq /= inv_freq.sum()
    weights = torch.from_numpy(inv_freq.astype(np.float32))
    return weights


def compute_iou_from_confusion_matrix(
    conf_matrix: np.ndarray, ignore_index: int
) -> Tuple[float, List[float]]:
    num_classes = conf_matrix.shape[0]
    ious: List[float] = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            iou = float("nan")
        else:
            iou = tp / denom
        ious.append(iou)
    # Mean over classes that have valid IoU
    valid_ious = [i for i in ious if not math.isnan(i)]
    if not valid_ious:
        return 0.0, ious
    return float(np.mean(valid_ious)), ious


def update_confusion_matrix(
    conf_matrix: np.ndarray,
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    """
    Update confusion matrix given predictions and labels.
    preds, labels: (N, H, W)
    """
    preds_np = preds.detach().cpu().numpy().astype(np.int64)
    labels_np = labels.detach().cpu().numpy().astype(np.int64)

    mask = labels_np != ignore_index
    preds_flat = preds_np[mask]
    labels_flat = labels_np[mask]

    k = (labels_flat >= 0) & (labels_flat < num_classes)
    labels_flat = labels_flat[k]
    preds_flat = preds_flat[k]

    if labels_flat.size == 0:
        return

    indices = labels_flat * num_classes + preds_flat
    cm = np.bincount(indices, minlength=num_classes * num_classes)
    conf_matrix += cm.reshape((num_classes, num_classes))


# -----------------------------
# Dataset
# -----------------------------

class OffroadSegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_processor: AutoImageProcessor,
        image_size: int = 512,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_processor = image_processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image with cv2 (BGR) and convert to RGB
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load mask (unchanged)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # Resize image and mask to 512x512
        img_resized = cv2.resize(
            img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask_raw,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Remap mask values to class IDs 0-9, others to IGNORE_INDEX
        mask_mapped = remap_mask_values(mask_resized)

        # Convert to PIL Image for image_processor
        img_pil = Image.fromarray(img_resized)

        encoded = self.image_processor(
            images=img_pil,
            segmentation_maps=mask_mapped,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)  # (C, H, W)
        labels = encoded["labels"].squeeze(0).long()  # (H, W)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# -----------------------------
# Training & Evaluation
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    num_batches = 0

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.float16):
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        num_batches += 1

        # Update IoU confusion matrix
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # (B, H, W)
            update_confusion_matrix(
                conf_matrix, preds, labels, num_classes=num_classes, ignore_index=ignore_index
            )

    avg_loss = running_loss / max(num_batches, 1)
    mean_iou, _ = compute_iou_from_confusion_matrix(conf_matrix, ignore_index=-1)
    return avg_loss, mean_iou


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    num_batches = 0

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast(dtype=torch.float16):
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = loss_fn(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=1)
            update_confusion_matrix(
                conf_matrix, preds, labels, num_classes=num_classes, ignore_index=ignore_index
            )

    avg_loss = running_loss / max(num_batches, 1)
    mean_iou, _ = compute_iou_from_confusion_matrix(conf_matrix, ignore_index=-1)
    return avg_loss, mean_iou


# -----------------------------
# Main
# -----------------------------

def main():
    maybe_mount_google_drive()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image processor
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Collect file paths
    train_image_paths, train_mask_paths = list_sorted_files(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
    val_image_paths, val_mask_paths = list_sorted_files(VAL_IMAGE_DIR, VAL_MASK_DIR)

    print(f"Train samples: {len(train_image_paths)}")
    print(f"Val samples:   {len(val_image_paths)}")

    if len(train_image_paths) == 0 or len(val_image_paths) == 0:
        raise RuntimeError("No images found. Check dataset paths in Google Drive.")

    # Compute class weights from training masks
    print("Computing class weights...")
    class_weights = compute_class_weights(train_mask_paths, NUM_CLASSES, IGNORE_INDEX)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")

    # Datasets & loaders
    train_dataset = OffroadSegmentationDataset(
        train_image_paths, train_mask_paths, image_processor, image_size=IMAGE_SIZE
    )
    val_dataset = OffroadSegmentationDataset(
        val_image_paths, val_mask_paths, image_processor, image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Loss, optimizer, scheduler
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_training_steps = len(train_loader) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler()

    best_iou = 0.0

    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            scaler,
            NUM_CLASSES,
            IGNORE_INDEX,
        )

        val_loss, val_iou = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            NUM_CLASSES,
            IGNORE_INDEX,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        torch.save(model.state_dict(), LAST_MODEL_PATH)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Best IoU: {best_iou:.4f}"
        )


if __name__ == "__main__":
    main()

