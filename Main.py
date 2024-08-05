import numpy as np
import random

from MLP import MLP
from MNIST.Donnees import Donnees
from TransfertFunction import Sigmoide
from TransfertFunction import Hyperbolique

# Example usage
if __name__ == "__main__":
    # Configure MLP
    nb_images = 10000
    pixels = 28 * 28  # Assuming MNIST images are 28x28 pixels
    layers = [pixels, 50, 50, 10]
    learning_rate = 0.6
    activation_function = Sigmoide()  # Choose activation function

    mlp = MLP(layers, learning_rate, activation_function)

    # Load MNIST data
    data = Donnees.load_imagette(
        nb_images,
        "donnees/images.idx3-ubyte",
        "donnees/labels.idx1-ubyte",
    )
    imagettes_array = data.get_imagettes_array()

    # Format training data 
    training_inputs = np.array(
        [imagette.get_imagette().flatten() for imagette in imagettes_array]
    )
    training_outputs = np.array(
        [imagette.get_etiquette() for imagette in imagettes_array]
    )
    # création d'un tableau de sortie one-hot
    # sert à transformer les étiquettes en vecteurs de sortie
    one_hot_outputs = np.zeros((training_outputs.size, training_outputs.max() + 1))
    # crée un vecteur de sortie one-hot pour chaque étiquette
    # sert à transformer les étiquettes en vecteurs de sortie
    one_hot_outputs[np.arange(training_outputs.size), training_outputs] = 1

    # Training loop
    max_epochs = 1000000
    target_error = 0.1
    epoch = 1
    total_error = 1.0

    while epoch < max_epochs and (total_error / epoch) > target_error:
        random_index = random.randint(0, nb_images - 1)
        input_data = training_inputs[random_index]
        output_data = one_hot_outputs[random_index]

        error = mlp.backPropagate(input_data, output_data)
        total_error += error
        epoch += 1

        print(f"Epoch: {epoch}, Error: {total_error / epoch}", end="\r")

    # Testing on training examples
    num_failed = 0
    print("Testing on training examples:")
    for i in range(len(training_inputs)):
        input_data = training_inputs[i]
        predicted_output = mlp.feed_forward(input_data)
        predicted_label = np.argmax(predicted_output)
        actual_label = training_outputs[i]
        print(f"Expected: {actual_label}, Predicted: {predicted_label}")
        if actual_label != predicted_label:
            num_failed += 1

    print(f"Failed: {num_failed}/{len(training_inputs)}")