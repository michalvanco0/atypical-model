import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from attacks.fgsm import fgsm_attack
from models.cnn_bn import SimpleCNN_BN



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN_BN().to(device)
    model.load_state_dict(torch.load("cnn_mnist_bn.pt", map_location=device))
    model.eval()

    correct_samples = torch.load("analysis_data/correct_high_conf_bn.pt")

    epsilon = 0.15
    adv_examples = []

    for sample in correct_samples:
        x = sample["image"].to(device)
        y = torch.tensor([sample["true_label"]], device=device)

        x_adv = fgsm_attack(model, x, y, epsilon)

        with torch.no_grad():
            logits = model(x_adv)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        if pred.item() != y.item():
            adv_examples.append({
                "original": x.cpu(),
                "adversarial": x_adv.cpu(),
                "true_label": y.item(),
                "adv_label": pred.item(),
                "confidence": conf.item(),
                "epsilon": epsilon
            })

        if len(adv_examples) >= 100:
            break

    os.makedirs("analysis_data", exist_ok=True)
    torch.save(adv_examples, "analysis_data/fgsm_adversarial_bn.pt")

    print(f"Saved {len(adv_examples)} FGSM adversarial samples")


if __name__ == "__main__":
    main()
