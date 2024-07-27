import struct

import numpy as np

class Etiquette:
    @staticmethod
    def get_etiquette(path):
        """
        Lit les étiquettes des fichiers MNIST.

        :param path: Le chemin vers le fichier des étiquettes.
        :return: Un tableau des étiquettes.
        """
        with open(path, 'rb') as file:
            magic_number, num_labels = struct.unpack('>II', file.read(8))
            etiquettes = np.fromfile(file, dtype=np.uint8)
            return etiquettes
