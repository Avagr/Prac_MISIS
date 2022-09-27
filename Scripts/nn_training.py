import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn
from tqdm.auto import tqdm


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def occurence_accuracy(pred, correct):
    return torch.argmax(pred, dim=1) == correct


def train_one_epoch(model, train_dataloader, criterion, optimizer, device="cuda:0", verbose=False) -> float:
    model.train()
    losses = []

    for feat, res in train_dataloader:
        feat, res = feat.to(device), res.to(device)
        optimizer.zero_grad()
        prediction = model(feat)
        loss = criterion(prediction, res)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(losses).item()


@torch.no_grad()
def predict(model, val_dataloader, criterion, device="cuda:0", verbose=False) -> (np.array, np.array):
    model.to(device)
    model.eval()
    losses = []
    predictions = []

    for feat, res in val_dataloader:
        feat, res = feat.to(device), res.to(device)
        prediction = model(feat)
        # accuracies.extend(r2_score(prediction.cpu(), res.cpu()).mean())
        losses.append(criterion(prediction, res).item())
        predictions.extend(prediction.tolist())

    return np.array(losses), predictions


def train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, device="cuda:0", n_epochs=10,
          scheduler=None, verbose=False, check_dir=None, save_every=None, model_name="Model",
          show_tqdm=False, wandb=False) -> (list[float], list[float]):
    model.to(device)
    last_test_pred = None
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, show_tqdm)
        val_losses, val_pred = predict(model, val_dataloader, criterion, device, show_tqdm)
        test_losses, test_pred = predict(model, test_dataloader, criterion, device, show_tqdm)
        last_test_pred = test_pred
        val_loss = np.mean(val_losses).item()
        test_loss = np.mean(test_losses).item()
        if wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f},"
                  f" Test loss: {test_loss:.4f}")
        if scheduler:
            scheduler.step(val_loss)
        if check_dir and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), check_dir + f"/{model_name}/state.e{epoch}_l{val_loss:.5f}")
    if check_dir:
        torch.save(model.state_dict(), check_dir + f"/{model_name}/state.fin_l{val_loss:.5f}")
    _, train_pred = predict(model, train_dataloader, criterion, device, show_tqdm)
    return train_pred, last_test_pred

