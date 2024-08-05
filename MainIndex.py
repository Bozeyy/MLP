import random
import time
import numpy as np
from MLP import MLP
from TransfertFunction import Sigmoide
from crypto.index import tabIndex


if __name__ == "__main__":
    # Configuration du MLP
    sequence_length = 99  # Longueur des séquences d'entrée
    layers = [sequence_length, 300, 300, 150, 2]  # Structure du MLP avec 2 neurones en sortie
    learning_rate = 0.3  # Taux d'apprentissage
    activation_function = Sigmoide()  # Fonction d'activation choisie (Sigmoide)

    # Initialisation du MLP avec les paramètres définis
    mlp = MLP(layers, learning_rate, activation_function)

    # Chargement des données de marché
    data = tabIndex()
    data.readFile("./crypto/bitPrice.csv")  # Lecture des données à partir d'un fichier CSV

    # Boucle d'entraînement
    max_epochs = 1000000  # Nombre maximum d'époques
    target_error = 0.1  # Erreur cible pour arrêter l'entraînement
    epoch = 1  # Initialisation du compteur d'époques
    total_error = 1.0  # Initialisation de l'erreur totale

    while epoch < max_epochs and (total_error / epoch) > target_error:
        # Obtention d'une séquence aléatoire
        random_sequence = data.getRandomSequence(sequence_length + 1)
        input_data = np.array(random_sequence[:-1])  # Séquence d'entrée sans le dernier élément
        target_value = random_sequence[-1]  # Dernier élément de la séquence aléatoire comme cible

        # Création de la sortie en format one-hot encoding
        # cette étape sert à transformer la valeur cible en un vecteur de sortie
        target_label = 1 if target_value > float(input_data[-1]) else 0
        output_data = np.zeros(2)
        output_data[target_label] = 1

        # Propagation arrière (backpropagation) pour ajuster les poids du réseau
        error = mlp.backPropagate(input_data, output_data)
        total_error += error  # Mise à jour de l'erreur totale
        epoch += 1  # Incrémentation de l'époque

        # Affichage de l'erreur moyenne pour l'époque actuelle
        print(f"Epoch: {epoch}, Error: {total_error / epoch}", end="\r")

    # Test sur les exemples d'entraînement
    num_failed = 0  # Initialisation du compteur d'échecs
    print("\nTesting on training examples:")
    sequences = data.getSequences(sequence_length)
    for sequence, target in sequences:
        input_data = np.array(sequence)
        predicted_output = mlp.feed_forward(input_data)  # Propagation avant pour obtenir la prédiction
        predicted_label = np.argmax(predicted_output)  # Obtention de l'étiquette prédite
        actual_label = 1 if target > sequence[-1] else 0  # Étiquette réelle
        print(f"Expected: {actual_label}, Predicted: {predicted_label}")
        if actual_label != predicted_label:
            num_failed += 1  # Incrémentation du compteur d'échecs si la prédiction est incorrecte

    # Affichage du nombre total d'échecs
    print(f"Failed: {num_failed}/{len(sequences)}")
    # Configuration du MLP
    sequence_length = 99  # Longueur des séquences d'entrée
    layers = [sequence_length, 50, 50, 2]  # Structure du MLP avec 2 neurones en sortie
    learning_rate = 0.3  # Taux d'apprentissage
    activation_function = Sigmoide()  # Fonction d'activation choisie (Sigmoide)

    # Initialisation du MLP avec les paramètres définis
    mlp = MLP(layers, learning_rate, activation_function)

    # Chargement des données de marché
    data = tabIndex()
    data.readFile("./crypto/bitPrice.csv")  # Lecture des données à partir d'un fichier CSV

    # Boucle d'entraînement
    max_epochs = 1000000  # Nombre maximum d'époques
    target_error = 0.1  # Erreur cible pour arrêter l'entraînement
    epoch = 1  # Initialisation du compteur d'époques
    total_error = 1.0  # Initialisation de l'erreur totale

    while epoch < max_epochs and (total_error / epoch) > target_error:
        # Obtention d'une séquence aléatoire
        random_sequence = data.getRandomSequence(sequence_length)
        input_data = np.array(random_sequence[:-1])  # Séquence d'entrée sans le dernier élément
        target_value = random_sequence[-1]  # Dernier élément de la séquence aléatoire comme cible

        # Création de la sortie en format one-hot encoding
        target_label = 1 if target_value > float(input_data[-1]) else 0
        output_data = np.zeros(2)
        output_data[target_label] = 1

        # Propagation arrière (backpropagation) pour ajuster les poids du réseau
        error = mlp.backPropagate(input_data, output_data)
        total_error += error  # Mise à jour de l'erreur totale
        epoch += 1  # Incrémentation de l'époque

        # Affichage de l'erreur moyenne pour l'époque actuelle
        print(f"Epoch: {epoch}, Error: {total_error / epoch}", end="\r")

    # Test sur les exemples d'entraînement
    num_failed = 0  # Initialisation du compteur d'échecs
    print("\nTesting on training examples:")
    sequences = data.getSequences(sequence_length)
    for sequence, target in sequences:
        input_data = np.array(sequence)
        predicted_output = mlp.feed_forward(input_data)  # Propagation avant pour obtenir la prédiction
        predicted_label = np.argmax(predicted_output)  # Obtention de l'étiquette prédite
        actual_label = 1 if target > sequence[-1] else 0  # Étiquette réelle
        print(f"Expected: {actual_label}, Predicted: {predicted_label}")
        if actual_label != predicted_label:
            num_failed += 1  # Incrémentation du compteur d'échecs si la prédiction est incorrecte

    # Affichage du nombre total d'échecs
    print(f"Failed: {num_failed}/{len(sequences)}")
