import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr
from tqdm import tqdm # Progress bar

from models import CompleteModel
from data_loader import get_data

LEARNING_RATE = 0.001
NUM_ITERATIONS = 3000
SCHEDULER_STEP_SIZE = 500
SCHEDULER_GAMMA = 0.5
EVAL_INTERVAL = 200

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, scores in test_loader:
            videos = videos.to(device)
            labels = scores.numpy()

            predictions = model(videos).cpu().numpy()

            all_preds.extend(predictions)
            all_labels.extend(labels)

    # Spearman Rank Correlation
    correlation, p_value = spearmanr(all_preds, all_labels)
    model.train()
    return correlation

def train_model(model, train_loader, test_loader, num_iterations, device):

    criterion = nn.MSELoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    train_iterator = iter(train_loader)

    print("Training...")
    for i in tqdm(range(num_iterations), desc="Training Iterations"):
        try:
            videos, scores = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            videos, scores = next(train_iterator)

        videos = videos.to(device)
        scores = scores.to(device)

        # Training
        optimizer.zero_grad()       # Clear old gradients
        predictions = model(videos) # Forward pass
        loss = criterion(predictions, scores) # Calculate loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update weights
        scheduler.step()            # Update learning rate

        # Show progress
        if (i + 1) % EVAL_INTERVAL == 0:
            correlation = evaluate_model(model, test_loader, device)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nIteration {i+1}/{num_iterations} | Loss: {loss.item():.4f} | Spearman Correlation: {correlation:.4f} | LR: {current_lr:.6f}")

    print("End")
    final_correlation = evaluate_model(model, test_loader, device)
    print(f"Spearman Correlation (test): {final_correlation:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    diving_data, vault_data = get_data(batch_size=1)
    print("Data loaded.")

    # --- Train Diving Model ---
    print("\n--- TRAINING DIVING MODEL ---")
    diving_model = CompleteModel().to(device)
    diving_model.load_pretrained_weights('c3d.pickle')
    for param in diving_model.c3d.parameters():
        param.requires_grad = False
    
    train_model(
        model=diving_model,
        train_loader=diving_data['train_loader'],
        test_loader=diving_data['test_loader'],
        num_iterations=NUM_ITERATIONS,
        device=device
    )
    torch.save(diving_model.state_dict(), 'diving_model_weights_2.pth')
    print("Diving model weights saved to diving_model_weights_2.pth")

    # --- Train Vault Model ---
    print("\n--- TRAINING GYMNASTIC VAULT MODEL ---")
    vault_model = CompleteModel().to(device)
    vault_model.load_pretrained_weights('c3d.pickle')
    for param in vault_model.c3d.parameters():
        param.requires_grad = False
    
    train_model(
        model=vault_model,
        train_loader=vault_data['train_loader'],
        test_loader=vault_data['test_loader'],
        num_iterations=NUM_ITERATIONS,
        device=device
    )
    torch.save(vault_model.state_dict(), 'vault_model_weights_2.pth')
    print("Vault model weights saved to vault_model_weights_2.pth")
