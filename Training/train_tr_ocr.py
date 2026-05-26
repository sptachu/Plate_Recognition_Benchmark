import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
from jiwer import cer

# --- 1. CONFIGURATION ---
TRAIN_CSV = "dataset/UC3M-LP/train_ocr/labels.csv"  # Your CSV file
IMAGE_DIR = "dataset/UC3M-LP/train_ocr/"       # Folder with your YOLO crops
MODEL_NAME = "microsoft/trocr-small-printed"
OUTPUT_DIR = "./trocr-license-plates"

# --- 2. DATASET CLASS ---
class LicensePlateDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor, max_target_length=10):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file name and text
        file_name = self.df.iloc[idx]['filename']
        text = self.df.iloc[idx]['words']
        
        # Load and prepare image
        image = Image.open(f"{self.img_dir}/{file_name}").convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Prepare text labels
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        # CRITICAL: Replace padding token id (usually 1 or 0) with -100
        # This tells PyTorch: "Don't calculate loss on these tokens"
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# --- 3. METRICS ---
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    # Calculate Character Error Rate (CER)
    cer_score = cer(label_str, pred_str)
    return {"cer": cer_score}

# --- 4. INITIALIZE MODEL & DATA ---
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Required settings for TrOCR
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Load dataset
train_dataset = LicensePlateDataset(TRAIN_CSV, IMAGE_DIR, processor)

# --- 5. TRAINING ARGUMENTS ---
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    per_device_train_batch_size=6,   # Adjust based on your GPU VRAM
    num_train_epochs=6,             # License plates learn quickly
    fp16=True,                       # Use Mixed Precision (faster, uses less memory)
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=10,              # Keep only the best 2 models
    learning_rate=2e-5,
)

# --- 6. START TRAINING ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)

print("[*] Starting training...")
trainer.train()

# Save the final model and processor together
trainer.save_model(f"{OUTPUT_DIR}/final_model")
processor.save_pretrained(f"{OUTPUT_DIR}/final_model")
print(f"[*] Training complete! Model saved to {OUTPUT_DIR}/final_model")