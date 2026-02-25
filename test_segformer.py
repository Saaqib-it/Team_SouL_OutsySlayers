import os
import glob

import cv2
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor


MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
NUM_CLASSES = 10
# Update these paths to match your Google Drive layout (hack folder)
MODEL_WEIGHTS_PATH = "/content/drive/MyDrive/hack/best_segformer.pth"
DATASET_DIR = "/content/drive/MyDrive/hack/Offroad_Segmentation_testImages/Color_Images"
OUTPUT_DIR = "/content/drive/MyDrive/hack/test_predictions"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device):
    """
    Load SegFormer model with the same architecture used for training,
    then load the fine-tuned weights from disk.
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    # Handle cases where the checkpoint may include additional metadata
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, processor


def run_inference_on_image(model, processor, image_path: str, device: torch.device) -> np.ndarray:
    """
    Run SegFormer inference on a single image and return the predicted mask
    resized back to the original image size.
    """
    # 1. Load image using cv2
    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_h, original_w = bgr_image.shape[:2]

    # 2. Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 3. Resize to 512x512
    rgb_resized = cv2.resize(rgb_image, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Prepare input for model
    inputs = processor(images=rgb_resized, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # 4. Run model inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # 5. Get predicted mask using argmax
    logits = outputs.logits  # (batch_size, num_classes, h, w)
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # 6. Resize mask back to original size
    mask_resized = cv2.resize(
        predicted,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask_resized


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_image_paths(directory: str):
    # Common image extensions; adjust if needed
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(image_paths)


def save_mask(mask: np.ndarray, original_image_path: str, output_dir: str):
    """
    Save the segmentation mask as a PNG file, preserving the base filename.
    """
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_mask.png")

    # Ensure mask is uint8
    mask_uint8 = mask.astype(np.uint8)

    # 7. Save mask as PNG
    cv2.imwrite(out_path, mask_uint8)


def main():
    device = get_device()
    print(f"Using device: {device}")

    ensure_output_dir(OUTPUT_DIR)
    model, processor = load_model(device)

    image_paths = get_image_paths(DATASET_DIR)
    if not image_paths:
        print(f"No images found in directory: {DATASET_DIR}")
        return

    print(f"Found {len(image_paths)} images. Running inference...")

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            mask = run_inference_on_image(model, processor, img_path, device)
            save_mask(mask, img_path, OUTPUT_DIR)
            print(f"[{idx}/{len(image_paths)}] Saved mask for: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("Inference complete. Masks saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

