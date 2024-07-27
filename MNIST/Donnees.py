import struct
import numpy as np
from MNIST.Etiquette import Etiquette
from MNIST.Imagette import Imagette

class Donnees:
    def __init__(self, imagettes_array, path):
        """
        Initialise une instance de la classe Donnees.

        :param imagettes_array: Tableau d'instances de la classe Imagette.
        :param path: Chemin vers le fichier des étiquettes.
        """
        etiquettes = Etiquette.get_etiquette(path)
        for i in range(len(imagettes_array)):
            imagettes_array[i].set_etiquette(etiquettes[i])
        self.imagettes_array = imagettes_array

    @staticmethod
    def load_imagette(nb_iteration_demande, path_image, path_etiquette):
        """
        Charge les images et leurs étiquettes depuis les fichiers MNIST.

        :param nb_iteration_demande: Nombre d'images à charger.
        :param path_image: Chemin vers le fichier des images.
        :param path_etiquette: Chemin vers le fichier des étiquettes.
        :return: Une instance de la classe Donnees.
        """
        with open(path_image, 'rb') as file:
            magic_number, nb_images, num_rows, num_cols = struct.unpack('>IIII', file.read(16))
            nb_iteration = nb_iteration_demande if nb_iteration_demande >= 0 else nb_images
            nb_iteration = min(nb_iteration, nb_images)
            
            imagettes = []
            for _ in range(nb_iteration):
                pixels = np.fromfile(file, dtype=np.uint8, count=num_rows*num_cols).reshape((num_rows, num_cols))
                imagettes.append(Imagette(pixels))
            
        return Donnees(imagettes, path_etiquette)
    
    def get_imagettes_array(self):
        """
        Retourne le tableau d'instances de la classe Imagette.

        :return: Tableau d'instances de la classe Imagette.
        """
        return self.imagettes_array
