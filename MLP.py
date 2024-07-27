from Layer import Layer


class MLP:
    def __init__(self, layers, learning_rate, fun):
        """
        Initialise le réseau de neurones multicouche (MLP).

        :param layers: Liste contenant le nombre de neurones par couche.
        :param learning_rate: Taux d'apprentissage.
        :param fun: Fonction de transfert.
        """
        self.fLearningRate = learning_rate
        self.fTransferFunction = fun
        self.fLayers = []

        for i in range(len(layers)):
            if i != 0:
                self.fLayers.append(Layer(layers[i], layers[i - 1]))
            else:
                self.fLayers.append(Layer(layers[i], 0))

    def execute(self, input):
        """
        Réponse à une entrée.

        :param input: L'entrée testée.
        :return: Résultat de l'exécution.
        """
        output = [0.0] * self.fLayers[-1].Length

        for i in range(self.fLayers[0].Length):
            self.fLayers[0].Neurons[i].Value = input[i]

        for k in range(1, len(self.fLayers)):
            for i in range(self.fLayers[k].Length):
                new_value = 0.0
                for j in range(self.fLayers[k - 1].Length):
                    new_value += self.fLayers[k].Neurons[i].Weights[j] * self.fLayers[k - 1].Neurons[j].Value

                new_value -= self.fLayers[k].Neurons[i].Bias
                self.fLayers[k].Neurons[i].Value = self.fTransferFunction.evaluate(new_value)

        for i in range(self.fLayers[-1].Length):
            output[i] = self.fLayers[-1].Neurons[i].Value

        return output

    def backPropagate(self, input, output):
        """
        Rétropropagation.

        :param input: L'entrée courante.
        :param output: Sortie souhaitée (apprentissage supervisé).
        :return: Erreur entre la sortie calculée et la sortie souhaitée.
        """
        new_output = self.execute(input)
        error = 0.0

        for i in range(len(self.fLayers[-1].Neurons)):
            error_val = output[i] - new_output[i]
            self.fLayers[-1].Neurons[i].Delta = error_val * self.fTransferFunction.evaluateDer(new_output[i])

        for k in range(len(self.fLayers) - 2, -1, -1):
            for i in range(self.fLayers[k].Length):
                error = 0.0
                for j in range(self.fLayers[k + 1].Length):
                    error += self.fLayers[k + 1].Neurons[j].Delta * self.fLayers[k + 1].Neurons[j].Weights[i]
                self.fLayers[k].Neurons[i].Delta = error * self.fTransferFunction.evaluateDer(self.fLayers[k].Neurons[i].Value)
            
            for i in range(self.fLayers[k + 1].Length):
                for j in range(self.fLayers[k].Length):
                    self.fLayers[k + 1].Neurons[i].Weights[j] += self.fLearningRate * self.fLayers[k + 1].Neurons[i].Delta * self.fLayers[k].Neurons[j].Value
                self.fLayers[k + 1].Neurons[i].Bias -= self.fLearningRate * self.fLayers[k + 1].Neurons[i].Delta

        error = sum(abs(new_output[i] - output[i]) for i in range(len(output))) / len(output)
        return error

    def getLearningRate(self):
        """
        Retourne le taux d'apprentissage.

        :return: Taux d'apprentissage.
        """
        return self.fLearningRate

    def setLearningRate(self, rate):
        """
        Met à jour le taux d'apprentissage.

        :param rate: Nouveau taux d'apprentissage.
        """
        self.fLearningRate = rate

    def setTransferFunction(self, fun):
        """
        Met à jour la fonction de transfert.

        :param fun: Nouvelle fonction de transfert.
        """
        self.fTransferFunction = fun

    def getInputLayerSize(self):
        """
        Retourne la taille de la couche d'entrée.

        :return: Taille de la couche d'entrée.
        """
        return self.fLayers[0].Length

    def getOutputLayerSize(self):
        """
        Retourne la taille de la couche de sortie.

        :return: Taille de la couche de sortie.
        """
        return self.fLayers[-1].Length
