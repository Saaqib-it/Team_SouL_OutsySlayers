Google Drive Link -- https://drive.google.com/drive/folders/1By6CmJ-Eddt8ErtBIKZvd2SP-ruehtS3?usp=drive_link


# Duality AI Hackathon Submission

## Team: Team SouL
## Final IoU Score: 0.5877

## How to Run

1. Install requirements:
   - Create and activate a virtual environment (optional but recommended):
     - `python -m venv .venv`
     - On Windows: `.\.venv\Scripts\activate`
     - On Linux/macOS/Colab: `source .venv/bin/activate`
   - Install Python dependencies:
     - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
     - `pip install transformers opencv-python pillow tqdm`

2. Activate environment:
   - If you created the `.venv` environment above, activate it before running any scripts:
     - Windows: `.\.venv\Scripts\activate`
     - Linux/macOS: `source .venv/bin/activate`
   - In Google Colab, this step is not required; just install the packages in a cell.

3. Train model:
   - Make sure the training dataset is placed under:
     - `/content/drive/MyDrive/hack/Offroad_Segmentation_Training_Dataset`
   - Run the training script:
     - `python train_segformer.py`
   - This trains a SegFormer-B2 model for 50 epochs and saves:
     - Best model: `/content/drive/MyDrive/hack/best_segformer.pth`
     - Last model: `/content/drive/MyDrive/hack/last_segformer.pth`

4. Test / evaluate model:
   - For semantic segmentation inference on test images:
     - Ensure test images are under:
       - `/content/drive/MyDrive/hack/Offroad_Segmentation_testImages/Color_Images`
     - Run:
       - `python test_segformer.py`
   - For quantitative evaluation on the validation set (IoU, Dice, Accuracy, Loss):
     - Use the evaluation script (for example `eval_segformer.py`) configured with:
       - Validation images: `/content/drive/MyDrive/hack/Offroad_Segmentation_Training_Dataset/val/Color_Images`
       - Validation masks: `/content/drive/MyDrive/hack/Offroad_Segmentation_Training_Dataset/val/Segmentation`
       - Model weights: `/content/drive/MyDrive/hack/best_segformer.pth`

## Model Details

- Architecture: `SegformerForSemanticSegmentation`
- Backbone: `nvidia/segformer-b2-finetuned-ade-512-512`
- Epochs: `50`
- Input Size: `512 x 512`
- Number of Classes: `10`
- Ignore Index: `255`
- Batch Size: `2` (training), `4` (evaluation)

