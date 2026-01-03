import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.mnist import get_mnist_loaders
from models.cnn import SimpleCNN


def train_one_epoch(model, loader, optimizer, device, lambda_gp=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        grad_x = torch.autograd.grad(
            loss, x, create_graph=True
        )[0]
        grad_penalty = grad_x.view(grad_x.size(0), -1).norm(2, dim=1).mean()

        total = loss + lambda_gp * grad_penalty
        total.backward()
        optimizer.step()

        total_loss += total.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, lambda_gp=0.1
        )

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={correct/total:.4f}")

    torch.save(model.state_dict(), "cnn_mnist_gp.pt")
    print("Saved cnn_mnist_gp.pt")


if __name__ == "__main__":
    main()
