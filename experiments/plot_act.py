import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt

# from models.cnn_bn import SimpleCNN_BN
from models.cnn import SimpleCNN
from analysis.activations import ActivationExtractor

def activation_distance(a, b):
    return torch.norm(a.flatten() - b.flatten()).item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = SimpleCNN().to(device)
    # model.load_state_dict(torch.load("cnn_mnist.pt", map_location=device))
    # model = SimpleCNN_BN().to(device)
    # model.load_state_dict(torch.load("cnn_mnist_bn.pt", map_location=device))
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_mnist_gp.pt", map_location=device))
    model.eval()

    # adv_data = torch.load("analysis_data/fgsm_adversarial.pt")
    # adv_data = torch.load("analysis_data/fgsm_adversarial_bn.pt")
    adv_data = torch.load("analysis_data/fgsm_adversarial_gp.pt")

    layers = ["conv1", "conv2", "fc1"]
    extractor = ActivationExtractor(model, layers)

    distances = {layer: [] for layer in layers}

    for sample in adv_data:
        x = sample["original"].to(device)
        x_adv = sample["adversarial"].to(device)

        extractor.clear()
        _ = model(x)
        acts_orig = extractor.activations.copy()

        extractor.clear()
        _ = model(x_adv)
        acts_adv = extractor.activations.copy()

        for layer in layers:
            d = activation_distance(
                acts_orig[layer], acts_adv[layer]
            )
            distances[layer].append(d)

    extractor.remove()

    avg_distances = [sum(distances[layer])/len(distances[layer]) for layer in layers]

    plt.figure(figsize=(6,4))
    plt.bar(layers, avg_distances, color="skyblue")
    plt.ylabel("Average Activation Distance")
    plt.title("Effect of FGSM Adversarial Perturbations by Layer")
    # plt.savefig("analysis_data/activation_distances.png")
    # plt.savefig("analysis_data/activation_distances_bn.png")
    plt.savefig("analysis_data/activation_distances_gp.png")
    plt.show()
    # print("Saved figure as analysis_data/activation_distances.png")
    # print("Saved figure as analysis_data/activation_distances_bn.png")
    print("Saved figure as analysis_data/activation_distances_gp.png")

if __name__ == "__main__":
    main()
