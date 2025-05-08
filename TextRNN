from google.colab import files
files.upload()  # Upload your kaggle.json here

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
!kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
!unzip -q ravdess-emotional-speech-audio.zip

# Step 1: Install Whisper and Other Dependencies
!pip install -q git+https://github.com/openai/whisper.git
!pip install -q torch torchvision torchaudio scikit-learn

import whisper
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter
import os
import glob

# Step 2: Load the Whisper Model
model_whisper = whisper.load_model("base")

# Step 3: Define Helper Functions to Transcribe Audio
def transcribe_audio(file):
    try:
        result = model_whisper.transcribe(file, fp16=False)
        return result["text"]
    except Exception as e:
        print(f"Error with {file}: {e}")
        return ""

# Step 4: Emotion Code Mapping
EMOTION_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

label_to_idx = {v: i for i, v in enumerate(sorted(set(EMOTION_LABELS.values())))}

# Step 5: Load Dataset and Extract Transcriptions
def extract_label(file_path):
    filename = os.path.basename(file_path)
    emotion_code = filename.split("-")[2]
    return EMOTION_LABELS[emotion_code]

def load_ravdess_file_paths(data_root=".", shuffle=True):
    paths = glob.glob(os.path.join(data_root, "Actor_*", "*.wav"))
    dataset = [(path, extract_label(path)) for path in paths]
    if shuffle:
        import random
        random.shuffle(dataset)
    return dataset

# Load the dataset
dataset = load_ravdess_file_paths()

# Step 6: Transcribe Audio Files and Collect Text and Labels
transcriptions, labels = [], []

for path, label in dataset:
    text = transcribe_audio(path)
    if text.strip():  # Only add non-empty transcriptions
        transcriptions.append(text)
        labels.append(label_to_idx[label])  # Use integer labels here


# Step 7: Preprocess Text Data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove non-alphanumeric characters
    return text.split()

# Build a vocabulary from the entire dataset
all_tokens = [token for t in transcriptions for token in preprocess_text(t)]
vocab = {"<PAD>": 0, "<UNK>": 1}
for word in Counter(all_tokens):
    vocab[word] = len(vocab)

# Step 8: Encode Text into Integer Sequences
def encode_text(text):
    return [vocab.get(w, vocab["<UNK>"]) for w in preprocess_text(text)]

# Step 9: Create Dataset Class for Text Data
class TextEmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encoded = [torch.tensor(encode_text(t)) for t in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]
X_train, X_val, y_train, y_val = train_test_split(transcriptions, labels, test_size=0.2, random_state=42)
train_dataset = TextEmotionDataset(X_train, y_train)
val_dataset = TextEmotionDataset(X_val, y_val)

# Create DataLoader
def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

# Step 11: Define the RNN Model
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # ×2 due to bidirection


    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        h = torch.cat((h[0], h[1]), dim=1)
        return self.fc(h.squeeze(0))

# Step 12: Set up Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextRNN(len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Step 13: Training and Evaluation Functions
def train(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Step 14: Training Loop
for epoch in range(40):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
