import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# =========================
# CONFIG
# =========================
DATASET_PATH = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\data\final_training_data_with_missing_classes.csv"

TEXT_COL = "text"
LABEL_COL = "label"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_gpu_output"
MODEL_SAVE_DIR = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_ats_model_gpu"
TOKENIZER_SAVE_DIR = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_ats_tokenizer_gpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TOKENIZER_SAVE_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATASET_PATH)

# Ensure correct columns exist
if LABEL_COL not in df.columns or TEXT_COL not in df.columns:
    raise ValueError(f"CSV must contain columns: '{LABEL_COL}' and '{TEXT_COL}'")

df = df.dropna(subset=[LABEL_COL, TEXT_COL])
df = df.drop_duplicates(subset=[TEXT_COL])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n==============================")
print("Original Class Distribution:")
print(df[LABEL_COL].value_counts())
print("==============================\n")

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df[LABEL_COL])

num_classes = len(label_encoder.classes_)
print("Classes Found:")
print(label_encoder.classes_)
print("Total Classes:", num_classes)
print("==============================\n")

# =========================
# SPLIT DATA (STRATIFIED)
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    df[TEXT_COL].tolist(),
    df["label_encoded"].tolist(),
    test_size=0.30,
    random_state=42,
    stratify=df["label_encoded"]
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.33,
    random_state=42,
    stratify=y_temp
)

print(f"Train: {len(X_train)}")
print(f"Val  : {len(X_val)}")
print(f"Test : {len(X_test)}")
print("==============================\n")

# =========================
# TOKENIZER
# =========================
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_function(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=MAX_LEN
    )

# =========================
# DATASET CLASS
# =========================
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN
        )
        enc["labels"] = self.labels[idx]
        return enc

train_dataset = EmailDataset(X_train, y_train)
val_dataset   = EmailDataset(X_val, y_val)
test_dataset  = EmailDataset(X_test, y_test)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# MODEL
# =========================
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes
)

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    report_to="none",

    fp16=torch.cuda.is_available(),  # mixed precision if GPU available
)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Training started...")
trainer.train()

# =========================
# EVALUATE
# =========================
print("\n✅ Evaluating on Test Set...")
preds = trainer.predict(test_dataset)

y_pred = np.argmax(preds.predictions, axis=1)

print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    digits=4
))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================
# SAVE MODEL + TOKENIZER
# =========================
print("\n💾 Saving model + tokenizer...")
trainer.model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)

# Save label encoder classes
label_path = os.path.join(MODEL_SAVE_DIR, "label_classes.txt")
with open(label_path, "w", encoding="utf-8") as f:
    for cls in label_encoder.classes_:
        f.write(cls + "\n")

print("\n✅ Done!")
print("Model saved at:", MODEL_SAVE_DIR)
print("Tokenizer saved at:", TOKENIZER_SAVE_DIR)
print("Label classes saved at:", label_path)

# =========================
# QUICK PREDICT FUNCTION
# =========================
def predict_email(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return {
        "predicted_class": label_encoder.classes_[pred_idx],
        "confidence": confidence
    }

print("\nSample Prediction:")
print(predict_email("Dear HR, I would like to know the onboarding process and joining date details."))
print("\n💾 FORCE SAVING MODEL + TOKENIZER...")

final_model_dir = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_ats_model_gpu"
final_tokenizer_dir = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_ats_tokenizer_gpu"

os.makedirs(final_model_dir, exist_ok=True)
os.makedirs(final_tokenizer_dir, exist_ok=True)

trainer.save_model(final_model_dir)          # saves model + config
tokenizer.save_pretrained(final_tokenizer_dir)

print("✅ Model saved to:", final_model_dir)
print("✅ Tokenizer saved to:", final_tokenizer_dir)

