import math


class TransferFunction:
    def evaluate(self, value):
        raise NotImplementedError("This method should be overridden by subclasses")

    def evaluateDer(self, value):
        raise NotImplementedError("This method should be overridden by subclasses")

    def getName(self):
        raise NotImplementedError("This method should be overridden by subclasses")


class Sigmoide(TransferFunction):
    def evaluate(self, value):
        """
        Évalue la fonction sigmoïde pour une valeur donnée.

        Fonction : σ(x) = 1 / (1 + e^(-x))

        :param value: La valeur d'entrée pour la fonction sigmoïde.
        :return: Le résultat de la fonction sigmoïde.
        """
        return 1 / (1 + math.exp(-value))

    def evaluateDer(self, value):
        """
        Évalue la dérivée de la fonction sigmoïde pour une valeur donnée.

        Dérivée : σ'(x) = σ(x) - σ^2(x)

        :param value: La valeur d'entrée pour la dérivée de la fonction sigmoïde.
        :return: Le résultat de la dérivée de la fonction sigmoïde.
        """
        return value - math.pow(value, 2)

    def getName(self):
        """
        Retourne le nom de la fonction de transfert.

        :return: Le nom de la fonction de transfert.
        """
        return "Sigmoide"


class Hyperbolique(TransferFunction):
    def evaluate(self, value):
        """
        Évalue la fonction hyperbolique tanh pour une valeur donnée.

        Fonction : σ(x) = tanh(x)

        :param value: La valeur d'entrée pour la fonction hyperbolique.
        :return: Le résultat de la fonction hyperbolique.
        """
        return math.tanh(value)

    def evaluateDer(self, value):
        """
        Évalue la dérivée de la fonction hyperbolique pour une valeur donnée.

        Dérivée : σ'(x) = 1 − σ^2(x)

        :param value: La valeur d'entrée pour la dérivée de la fonction hyperbolique.
        :return: Le résultat de la dérivée de la fonction hyperbolique.
        """
        return 1 - math.pow(value, 2)

    def getName(self):
        """
        Retourne le nom de la fonction de transfert.

        :return: Le nom de la fonction de transfert.
        """
        return "Hyperbolique"
