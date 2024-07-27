import random

class Neuron:
    def __init__(self, prev_layer_size):
        # La valeur de sortie du neurone après l'application de la fonction d'activation.
        self.Value = random.random() / 10000000000000.0
        
        # Liste des poids associés aux connexions entre ce neurone et les neurones de la couche précédente.
        # Chaque poids est initialisé à une valeur aléatoire.
        self.Weights = [random.random() / prev_layer_size for _ in range(prev_layer_size)]
        
        # Biais du neurone, initialisé à une valeur aléatoire.
        # Le biais est ajouté à la somme pondérée des entrées avant l'application de la fonction d'activation.
        self.Bias = random.random()
        
        # Delta est utilisé dans l'algorithme de rétropropagation pour la mise à jour des poids.
        # Il représente l'erreur associée au neurone.
        self.Delta = random.random() / 10000000000000.0