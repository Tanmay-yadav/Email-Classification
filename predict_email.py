import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# =========================
# PATHS (your saved model)
# =========================
MODEL_DIR = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_ats_model_gpu"
TOKENIZER_DIR = r"C:\Users\gupta\OneDrive\Desktop\email-generetaion-pipeline\Email-generation-pipelines\model_training\bert_ats_tokenizer_gpu"
LABEL_FILE = os.path.join(MODEL_DIR, "label_classes.txt")

MAX_LEN = 256

# =========================
# LOAD LABELS
# =========================
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"label_classes.txt not found at: {LABEL_FILE}")

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

print("✅ Loaded Labels:", labels)

# =========================
# LOAD MODEL + TOKENIZER
# =========================
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ Using device:", device)

# =========================
# PREDICT FUNCTION
# =========================
def predict_email(text, threshold=0.70):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()
    predicted_class = labels[pred_idx]

    result = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    }

    if confidence < threshold:
        result["warning"] = "Low confidence prediction"

    return result

# =========================
# TEST INPUTS
# =========================
if __name__ == "__main__":
    print("\n==============================")
    print("📩 ATS Email Classifier (BERT)")
    print("==============================\n")

    while True:
        user_text = input("Enter an email text (or type 'exit'): ").strip()
        if user_text.lower() == "exit":
            print("👋 Exiting...")
            break

        output = predict_email(user_text)
        print("\n✅ Prediction:", output)
        print("-" * 60)
