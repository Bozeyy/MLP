import numpy as np
from PIL import Image

class Imagette:
    def __init__(self, imagette):
        """
        Initialise une instance de la classe Imagette.

        :param imagette: Matrice 2D représentant l'image en niveaux de gris.
        """
        self.imagette = np.array(imagette)
        self.width = self.imagette.shape[0]
        self.height = self.imagette.shape[1]
        self.etiquette = None

    def set_etiquette(self, etiquette):
        """
        Définit l'étiquette associée à cette imagette.

        :param etiquette: L'étiquette de l'image.
        """
        self.etiquette = etiquette

    def screen(self, path):
        """
        Sauvegarde l'image sur le disque à l'emplacement spécifié.

        :param path: Le chemin où l'image sera sauvegardée.
        """
        image = Image.fromarray(self.imagette.astype('uint8'), 'L')
        image.save(path)
        print(f"Image sauvegardée avec succès : {path}")

    def get_width(self):
        """
        Retourne la largeur de l'image.

        :return: La largeur de l'image.
        """
        return self.width

    def get_height(self):
        """
        Retourne la hauteur de l'image.

        :return: La hauteur de l'image.
        """
        return self.height

    def get_imagette(self):
        """
        Retourne la matrice de pixels de l'image.

        :return: La matrice de pixels de l'image.
        """
        return self.imagette

    def get_etiquette(self):
        """
        Retourne l'étiquette de l'image.

        :return: L'étiquette de l'image.
        """
        return self.etiquette
    
    def draw(self):
        """
        Affiche l'image dans une fenêtre.
        """
        image = Image.fromarray(self.imagette.astype('uint8'), 'L')
        image.show()
