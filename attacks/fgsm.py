import torch
import torch.nn as nn


def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)

    model.zero_grad()
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    perturbation = epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv + perturbation, 0, 1)

    return x_adv.detach()
