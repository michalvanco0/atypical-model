import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from data_utils.mnist import get_mnist_loaders
from models.cnn_bn import SimpleCNN_BN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_mnist_loaders(batch_size=1)

    model = SimpleCNN_BN().to(device)
    model.load_state_dict(torch.load("cnn_mnist_bn.pt", map_location=device))
    model.eval()

    correct_high_conf = []
    wrong_high_conf = []

    CONF_THRESHOLD = 0.9
    MAX_SAMPLES = 100

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            record = {
                "image": x.cpu(),
                "true_label": y.item(),
                "pred_label": pred.item(),
                "confidence": conf.item()
            }

            if pred == y and conf > CONF_THRESHOLD:
                correct_high_conf.append(record)

            if pred != y and conf > CONF_THRESHOLD:
                wrong_high_conf.append(record)

            if len(correct_high_conf) >= MAX_SAMPLES and len(wrong_high_conf) >= MAX_SAMPLES:
                break

    os.makedirs("analysis_data", exist_ok=True)

    torch.save(correct_high_conf, "analysis_data/correct_high_conf_bn.pt")
    torch.save(wrong_high_conf, "analysis_data/wrong_high_conf_bn.pt")

    print(f"Saved {len(correct_high_conf)} high-confidence correct samples")
    print(f"Saved {len(wrong_high_conf)} high-confidence wrong samples")


if __name__ == "__main__":
    main()
