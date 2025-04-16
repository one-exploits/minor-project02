import os
import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ========== CONFIGURATION ==========
DATA_DIR = "/path/to/your/edf/files"  # 🔁 Replace with your actual path
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']  # Add more if needed
EPOCH_DURATION = 30  # seconds
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

# ========== DATASET LOADER ==========
class SleepEDFLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.psg_files = [f for f in os.listdir(data_dir) if "PSG.edf" in f]

    def load_all(self):
        all_X, all_y = [], []
        for psg_file in tqdm(self.psg_files, desc="Loading EDF pairs"):
            base_id = psg_file.split("-")[0]
            hyp_file = base_id + "EC-Hypnogram.edf"
            psg_path = os.path.join(self.data_dir, psg_file)
            hyp_path = os.path.join(self.data_dir, hyp_file)
            if not os.path.exists(hyp_path):
                continue
            try:
                X, y = self.load_pair(psg_path, hyp_path)
                all_X.append(X)
                all_y.extend(y)
            except Exception as e:
                print(f"Failed to load {base_id}: {e}")
        X = np.vstack(all_X)
        le = LabelEncoder()
        y = le.fit_transform(all_y)
        return X, y, le

    def load_pair(self, psg_path, hypnogram_path):
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        raw.pick_channels(EEG_CHANNELS)
        raw.resample(100)

        annots = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annots)

        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0,
                            tmax=EPOCH_DURATION - 1. / raw.info['sfreq'],
                            baseline=None, detrend=1, preload=True, verbose=False)
        data = epochs.get_data()
        labels = [raw.annotations.description[i] for i in range(len(epochs))]
        return data, labels

# ========== CUSTOM DATASET ==========
class SleepDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ========== CNN MODEL ==========
class SleepCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SleepCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ========== TRAINING SCRIPT ==========
def train():
    loader = SleepEDFLoader(DATA_DIR)
    X, y, label_encoder = loader.load_all()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(SleepDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SleepDataset(X_test, y_test), batch_size=BATCH_SIZE)

    input_size = X.shape[2]
    num_classes = len(label_encoder.classes_)
    model = SleepCNN(input_size=input_size, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # Test Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train()