import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from models import CompleteModel
from data_loader import get_data

DIVING_MODEL_PATH = 'diving_model_weights_2.pth'
VAULT_MODEL_PATH = 'vault_model_weights_2.pth'

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
    model.train() # Set back to train mode
    return correlation

def plot_predictions_vs_truth(model, test_loader, device, title, mean, std):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, scores in test_loader:
            videos = videos.to(device)

            predictions_normalized = model(videos).cpu().numpy()

            predictions_unnormalized = (predictions_normalized * std) + mean
            labels_unnormalized = (scores.numpy() * std) + mean

            all_preds.extend(predictions_unnormalized)
            all_labels.extend(labels_unnormalized)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.6, label='Predicciones vs Labels')

    plt.xlabel("Valores reales", fontsize=14)
    plt.ylabel("Predicciones", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.show()

def predict_single_video(model, dataset, video_index, device, mean, std):
    model.eval()

    video_tensor, true_normalized_score = dataset[video_index]

    video_tensor = video_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_normalized_score = model(video_tensor).cpu().item()

    predicted_score = (predicted_normalized_score * std) + mean
    true_score = (true_normalized_score.item() * std) + mean

    return true_score, predicted_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data for evaluation...")
    diving_data, vault_data = get_data(batch_size=1)
    
    diving_test_loader = diving_data['test_loader']
    diving_test_dataset = diving_data['test_dataset']
    diving_mean = diving_data['mean']
    diving_std = diving_data['std']
    
    vault_test_loader = vault_data['test_loader']
    vault_test_dataset = vault_data['test_dataset']
    vault_mean = vault_data['mean']
    vault_std = vault_data['std']

    print("Data loaded.")
    
    # --- Evaluate Diving Model ---
    print("\n--- EVALUATING DIVING MODEL ---")
    diving_model = CompleteModel().to(device)
    diving_model.load_state_dict(torch.load(DIVING_MODEL_PATH, map_location=device))
    diving_model.eval()
    print(f"Loaded model from: {DIVING_MODEL_PATH}")

    diving_corr = evaluate_model(diving_model, diving_test_loader, device)
    print(f"Diving Test Spearman Correlation: {diving_corr:.4f}")

    plot_predictions_vs_truth(
        model=diving_model,
        test_loader=diving_test_loader,
        device=device,
        title="Predicciones de los datos de Diving",
        mean=diving_mean,
        std=diving_std
    )

    # --- Evaluate Vault Model ---
    print("\n--- EVALUATING GYMNASTIC VAULT MODEL ---")
    vault_model = CompleteModel().to(device)
    vault_model.load_state_dict(torch.load(VAULT_MODEL_PATH, map_location=device))
    vault_model.eval()
    print(f"Loaded model from: {VAULT_MODEL_PATH}")
    
    vault_corr = evaluate_model(vault_model, vault_test_loader, device)
    print(f"Vault Test Spearman Correlation: {vault_corr:.4f}")

    plot_predictions_vs_truth(
        model=vault_model,
        test_loader=vault_test_loader,
        device=device,
        title="Predicciones de los datos de Vault",
        mean=vault_mean,
        std=vault_std
    )
    
    # --- Single Video Predictions ---
    print("\n--- SINGLE VIDEO PREDICTION EXAMPLES ---")
    true_score_1, pred_score_1 = predict_single_video(diving_model, diving_test_dataset, 22, device, diving_mean, diving_std)
    true_score_4, pred_score_4 = predict_single_video(diving_model, diving_test_dataset, 12, device, diving_mean, diving_std)
    true_score_2, pred_score_2 = predict_single_video(vault_model, vault_test_dataset, 45, device, vault_mean, vault_std)
    true_score_3, pred_score_3 = predict_single_video(vault_model, vault_test_dataset, 34, device, vault_mean, vault_std)

    print(f"Diving Example 1 (test sample #22): True Score = {true_score_1:.2f}, Predicted Score = {pred_score_1:.2f}")
    print(f"Diving Example 2 (test sample #12): True Score = {true_score_4:.2f}, Predicted Score = {pred_score_4:.2f}")
    print(f"Vault Example 1 (test sample #45): True Score = {true_score_2:.2f}, Predicted Score = {pred_score_2:.2f}")
    print(f"Vault Example 2 (test sample #34): True Score = {true_score_3:.2f}, Predicted Score = {pred_score_3:.2f}")
