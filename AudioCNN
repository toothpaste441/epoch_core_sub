class AudioCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioCNN, self).__init__()

        # Conv Block 1: 3 → 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Conv Block 2: 16 → 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Conv Block 3: 32 → 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Conv Block 4: 64 → 128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)


        # MaxPool (shared)
        self.pool = nn.MaxPool2d(2, 2)

        # Infer flattened size
        self._to_linear = None
        self._get_conv_output_shape()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output_shape(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 128, 256)  # Input shape with 3 channels
            x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 → 16 channels
            x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 → 32 channels
            x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 → 64 channels
            x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 → 128 channels

            self._to_linear = x.numel()  # Flattened output shape after Conv3

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 → 16 channels
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 → 32 channels
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 → 64 channels
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.dropout(F.relu(self.fc1(x)))  # Dropout for regularization
        return self.fc2(x)  # Output layer (classification)


# ---- TRAINING FUNCTION ----
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# ---- VALIDATION FUNCTION ----
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return val_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# ---- MAIN ----
if __name__ == '__main__':
    dataset = load_ravdess_file_paths("/content")  # or your RAVDESS path
    audio_dataset = RAVDESSAudioDataset(dataset)

    train_size = int(0.8 * len(audio_dataset))
    val_size = len(audio_dataset) - train_size
    train_set, val_set = random_split(audio_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
