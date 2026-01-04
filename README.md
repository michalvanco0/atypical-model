# Explaining Atypical Neural Network Behavior via Adversarial Activation Analysis

This project investigates cases in which a trained neural network exhibits atypical or unexpected behavior, such as high-confidence misclassifications and sensitivity to adversarial perturbations. Rather than focusing solely on output-level performance metrics, the project analyzes internal model activations to better understand how adversarial inputs affect different layers of the network.

Using a convolutional neural network trained on the MNIST dataset, adversarial examples are generated with the Fast Gradient Sign Method (FGSM). Layer-wise activation distances between original and adversarial inputs are then examined to identify where adversarial effects emerge and how they propagate through the model. Several mitigation strategies, including batch normalization and gradient-based regularization, are evaluated to study how architectural and training-time modifications influence internal adversarial sensitivity.

This project was developed as part of a Machine Learning course at Comenius University in Bratislava.

