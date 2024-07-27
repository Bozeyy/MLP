# import Neuron from Neuron.py
from Neuron import Neuron 

class Layer:
    def __init__(self, l, prev):
        """
        Couche de Neurones

        :param l: Taille de la couche
        :param prev: Taille de la couche précédente
        """
        # Longueur de la couche, c'est-à-dire le nombre de neurones dans cette couche.
        self.Length = l
        
        # Liste de neurones dans la couche, initialisée avec l neurones.
        # Chaque neurone est connecté à tous les neurones de la couche précédente (de taille prev).
        self.Neurons = [Neuron(prev) for _ in range(l)]