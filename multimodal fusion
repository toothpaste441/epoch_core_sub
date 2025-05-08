import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# STEP 1: Feature extraction function
def extract_features(file_path, max_len=160):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate to fixed length
    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_len]

    # Additional features (mean pitch, jitter, etc.)
    pitch = librosa.yin(y, fmin=50, fmax=300).mean()
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

    features = np.array([pitch, rms, zcr, centroid, bandwidth])
    
    return mel_spec_db, features

# STEP 2: Load and label data
import os

def load_data(data_path):
    X_spec, X_feats, y = [], [], []
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        # Now loop through each .wav file in the actor's folder
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                try:
                    emotion_code = file.split('-')[2]  # Extract emotion code from filename
                    emotion = emotion_map.get(emotion_code, None)  # Map emotion code to name
                    if emotion is None:
                        continue  # Skip if emotion code is not valid

                    # Get full path to the audio file
                    full_path = os.path.join(actor_path, file)
                    spec, feats = extract_features(full_path)  # Get spectrogram and additional features
                    
                    X_spec.append(spec)
                    X_feats.append(feats)
                    y.append(emotion)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
    
    return X_spec, X_feats, y




class MultimodalDataset(Dataset):
    def __init__(self, X_spec, X_feats, y):
        self.X_spec = torch.tensor(X_spec, dtype=torch.float32)  # Convert once
        self.X_feats = torch.tensor(X_feats, dtype=torch.float32)  # Convert once
        self.y = torch.tensor(y, dtype=torch.long)  # Convert once

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = self.X_spec[idx].unsqueeze(0)  # Add channel dimension [1, H, W] 
        feats = self.X_feats[idx]  # No need to create a new tensor here
        label = self.y[idx]  # No need to create a new tensor here
        return spec, feats, label


# STEP 4: CNN + Early Fusion Model
class EarlyFusionCNN(nn.Module):
    def __init__(self, cnn_flat_dim, feature_input_dim, num_classes):
        super(EarlyFusionCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_cnn = nn.Linear(cnn_flat_dim, 128)
        self.fc_feats = nn.Linear(feature_input_dim, 64)
        self.classifier = nn.Linear(128 + 64, num_classes)

    def forward(self, x_spec, x_feats):
        x = self.cnn(x_spec)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc_cnn(x))
        x_feats = F.relu(self.fc_feats(x_feats))
        x = torch.cat((x, x_feats), dim=1)
        return self.classifier(x)

# STEP 5: Training
X_spec, X_feats, y = load_data("/content")  # Update this to match the root path of your dataset
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, feats_train, feats_test, y_train, y_test = train_test_split(X_spec, X_feats, y_encoded, test_size=0.2)

train_ds = MultimodalDataset(X_train, feats_train, y_train)
test_ds = MultimodalDataset(X_test, feats_test, y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# Temporarily create CNN to estimate output shape
dummy_cnn = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

# Use one spectrogram sample to infer output shape
sample_input = torch.tensor(X_spec[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 128, 160]
with torch.no_grad():
    out = dummy_cnn(sample_input)
cnn_flat_dim = out.view(-1).shape[0]

feature_input_dim = torch.tensor(X_feats[0]).shape[0]
model = EarlyFusionCNN(cnn_flat_dim, feature_input_dim, num_classes=8)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def accuracy(preds, labels):
    _, pred_classes = torch.max(preds, 1)
    correct = (pred_classes == labels).sum().item()
    return correct / len(labels)


for epoch in range(25):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for spec, feats, labels in train_loader:
        optimizer.zero_grad()
        output = model(spec, feats)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += accuracy(output, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Validation
    model.eval()
    val_acc = 0.0
    with torch.no_grad():
        for spec, feats, labels in test_loader:
            output = model(spec, feats)
            val_acc += accuracy(output, labels)
    val_acc /= len(test_loader)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
