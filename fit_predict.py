import torch
import torch.nn as nn

def fit(model, loss_model, X, y, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # обнуляем веса
    optimizer.zero_grad()

    # forward + backward + optimize
    y_pred = model(X)
    loss = loss_model(y_pred, y)
    loss.backward()
    optimizer.step()

def predict(model, X):
    return torch.max(model(X), 1)
