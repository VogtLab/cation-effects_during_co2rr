import torch
import numpy as np
from sklearn.metrics import mean_squared_error

from model import get_device


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, device=None, verbose=True):
    if device is None:
        device = get_device()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, cations, targets in train_loader:
            inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, cations)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, cations, targets in val_loader:
                inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)
                outputs = model(inputs, cations)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = get_device()

    model.eval()
    all_y_pred = []
    all_y_true = []

    with torch.no_grad():
        for inputs, combined_encodings, targets in test_loader:
            inputs = inputs.to(device)
            combined_encodings = combined_encodings.to(device)
            targets = targets.to(device)

            outputs = model(inputs, combined_encodings)

            all_y_pred.append(outputs.cpu().numpy())
            all_y_true.append(targets.cpu().numpy())

    y_pred = np.vstack(all_y_pred).flatten()
    y_true = np.vstack(all_y_true).flatten()

    return y_pred, y_true


def calculate_feature_importance(model, X_test, combined_test_tensor, y_true, feature_peaks, time_steps, device=None):
    if device is None:
        device = get_device()

    feature_importance = np.zeros(len(feature_peaks))

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_pred_baseline = model(X_test_tensor, combined_test_tensor.to(device))
        y_pred_baseline = y_pred_baseline.cpu().numpy().flatten()

    baseline_mse = mean_squared_error(y_true, y_pred_baseline)

    for feature_idx, feature_name in enumerate(feature_peaks):
        X_permuted = X_test.copy()

        for t in range(time_steps):
            orig_values = X_permuted[:, t, feature_idx].copy()

            np.random.shuffle(X_permuted[:, t, feature_idx])

            X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)

            with torch.no_grad():
                y_permuted_pred = model(X_permuted_tensor, combined_test_tensor.to(device))
                y_permuted_pred = y_permuted_pred.cpu().numpy().flatten()

            permuted_mse = mean_squared_error(y_true, y_permuted_pred)

            feature_importance[feature_idx] += (permuted_mse - baseline_mse)

            X_permuted[:, t, feature_idx] = orig_values

    feature_importance /= time_steps

    return feature_importance
