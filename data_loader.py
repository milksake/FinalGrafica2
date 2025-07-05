import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat

DIVING_DIR = 'diving'
VAULT_DIR = 'gymnastic_vault'

DIVING_VIDEO_DIR = os.path.join(DIVING_DIR, 'diving_samples_len_151_lstm')
VAULT_VIDEO_DIR = os.path.join(VAULT_DIR, 'gym_vault_samples_len_100_lstm')

def read_video(video_path): # List of frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path}")
        return frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

video_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process(action_name, base_dir, video_dir, scores_file, train_idx_file, test_idx_file): # Diving or Gymnastics
    print(f"Processing {action_name}...")

    # Load .mat files
    scores_mat = loadmat(os.path.join(base_dir, scores_file))
    train_idx_mat = loadmat(os.path.join(base_dir, train_idx_file))
    test_idx_mat = loadmat(os.path.join(base_dir, test_idx_file))

    # Extract data
    scores = scores_mat['overall_scores'].flatten()
    train_indices = train_idx_mat['training_idx'].flatten()
    test_indices = test_idx_mat['testing_idx'].flatten()

    print(f"Total: {len(scores)}")
    print(f"Training: {len(train_indices)}")
    print(f"Testing: {len(test_indices)}")

    train_samples = []
    test_samples = []

    for i in train_indices:
        idx = i - 1
        video_path = os.path.join(video_dir, f"{i:03d}.avi")
        train_samples.append((video_path, scores[idx]))

    for i in test_indices:
        idx = i - 1
        video_path = os.path.join(video_dir, f"{i:03d}.avi")
        test_samples.append((video_path, scores[idx]))

    return train_samples, test_samples

class AqaDataset(Dataset):
    def __init__(self, samples, mean, std, transform=None):
        # samples: [(video_path, score)]
        self.samples = samples
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, score = self.samples[idx]

        frames = read_video(video_path)

        if self.transform:
            transformed_frames = [self.transform(frame) for frame in frames]
            video_tensor = torch.stack(transformed_frames)
        else:
            video_tensor = torch.from_numpy(np.array(frames))

        # Normalize
        normalized_score = (score - self.mean) / self.std
        score_tensor = torch.tensor(normalized_score, dtype=torch.float32)

        return video_tensor, score_tensor

def get_data(batch_size=1):
    # Diving data
    diving_train, diving_test = process(
        'Diving', DIVING_DIR, DIVING_VIDEO_DIR,
        'diving_overall_scores.mat', 'split_300_70/training_idx.mat', 'split_300_70/testing_idx.mat'
    )
    # Vault data
    vault_train, vault_test = process(
        'Gymnastic Vault', VAULT_DIR, VAULT_VIDEO_DIR,
        'gym_vault_overall_scores.mat', 'split_4/training_idx.mat', 'split_4/testing_idx.mat'
    )

    # Mean and STD (TRAINING ONLY)
    diving_train_scores = [score for path, score in diving_train]
    diving_mean = np.mean(diving_train_scores)
    diving_std = np.std(diving_train_scores)
    vault_train_scores = [score for path, score in vault_train]
    vault_mean = np.mean(vault_train_scores)
    vault_std = np.std(vault_train_scores)
    
    print(f"Diving: mean={diving_mean:.2f}, std={diving_std:.2f}")
    print(f"Vault: mean={vault_mean:.2f}, std={vault_std:.2f}")

    # Create Datasets
    diving_train_dataset = AqaDataset(diving_train, diving_mean, diving_std, transform=video_transform)
    diving_test_dataset = AqaDataset(diving_test, diving_mean, diving_std, transform=video_transform)
    vault_train_dataset = AqaDataset(vault_train, vault_mean, vault_std, transform=video_transform)
    vault_test_dataset = AqaDataset(vault_test, vault_mean, vault_std, transform=video_transform)

    # Create DataLoaders
    diving_train_loader = DataLoader(diving_train_dataset, batch_size=batch_size, shuffle=True)
    diving_test_loader = DataLoader(diving_test_dataset, batch_size=batch_size, shuffle=False)
    vault_train_loader = DataLoader(vault_train_dataset, batch_size=batch_size, shuffle=True)
    vault_test_loader = DataLoader(vault_test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Diving: {len(diving_train_dataset)} train, {len(diving_test_dataset)} test")
    print(f"Vault: {len(vault_train_dataset)} train, {len(vault_test_dataset)} test")

    diving_data = {
        'train_loader': diving_train_loader,
        'test_loader': diving_test_loader,
        'test_dataset': diving_test_dataset,
        'mean': diving_mean,
        'std': diving_std
    }
    vault_data = {
        'train_loader': vault_train_loader,
        'test_loader': vault_test_loader,
        'test_dataset': vault_test_dataset,
        'mean': vault_mean,
        'std': vault_std
    }
    
    return diving_data, vault_data
