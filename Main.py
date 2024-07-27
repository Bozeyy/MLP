from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np

from MLP import MLP
from MNIST.Donnees import Donnees
from TransfertFunction import Sigmoide
import time


def main(args):
    couches = []
    try:
        couches = [int(x) for x in args[0].split(',')]
        couches.append(10)
        initialLearningRate = 0.3
        finalLearningRate = 0.3
        shuffle = True
        tauxSortie = 0.05
        maxIteration = 40
    except Exception as e:
        print("Erreur dans la saisie des paramètres => ")
        print("Usage : python main.py couchesCachés initialLearningRate finalLearningRate schuffle tauxSortie maxIteration")
        print(e)
        exit(1)

    print("Chargement des données..")
    with ThreadPoolExecutor() as executor:
        future_train = executor.submit(Donnees.load_imagette, -1, 'donnees/images.idx3-ubyte', 'donnees/labels.idx1-ubyte')
        future_test = executor.submit(Donnees.load_imagette, -1, 'donnees/images.idx3-ubyte', 'donnees/labels.idx1-ubyte')
        donneesEntrainement = future_train.result()
        donneesTest = future_test.result()

    if donneesEntrainement is None or donneesTest is None:
        print("Impossible de lire les images")
        exit(1)

    print(f"nb images d'entrainement : {len(donneesEntrainement.get_imagettes_array())}")
    tailleImage = donneesEntrainement.get_imagettes_array()[0].get_height() * donneesEntrainement.get_imagettes_array()[0].get_width()
    couches.insert(0, tailleImage)

    mlp = MLP(couches, initialLearningRate, Sigmoide())

    imagesEntrainements = donneesEntrainement.get_imagettes_array()
    imagesDeTests = donneesTest.get_imagettes_array()

    runMLP(couches, initialLearningRate, finalLearningRate, shuffle, imagesEntrainements, imagesDeTests, tauxSortie, maxIteration, mlp)


def runMLP(couches, initialLearningRate, finalLearningRate, shuffle, imagesEntrainements, imagesDeTests, tauxSortie, maxIteration, mlp):
    iteration = 0
    learned = False
    leanringRateDegressif = (initialLearningRate != finalLearningRate)
    slope = (finalLearningRate - initialLearningRate) / maxIteration
    nbRepetition = 10
    allTime = time.time()
    
    print("Iteration;Taux d'erreur;Erreur moyenne;Durée" + (f";Taux d'apprentissage" if leanringRateDegressif else ""))

    while iteration < maxIteration and not learned:
        print(f"Iteration {iteration}")
        if leanringRateDegressif:
            mlp.setLearningRate(initialLearningRate + slope * iteration)
        startTime = time.time()
        errorSum = 0
        for rep in range(nbRepetition):
            print(f"  Répétition {rep}")
            if shuffle:
                random.shuffle(imagesEntrainements)
            for index, imagette in enumerate(imagesEntrainements):
                if index % 1000 == 0:
                    print(f"    Traitement de l'image {index}")
                etiquetteInput = imagette.get_etiquette()
                errorSum += mlp.backPropagate(toOneArray(imagette.get_imagette()), intToArray(int(etiquetteInput)))
        errorBackPropagate = errorSum / (len(imagesEntrainements) * 10)

        iteration += 1
        nbIncorrect = 0
        for imagette in imagesDeTests:
            etiquetteInput = imagette.get_etiquette()
            output = mlp.execute(toOneArray(imagette.get_imagette()))
            if findMaxIndex(output) != int(etiquetteInput):
                nbIncorrect += 1
            print(f"Attendu : {etiquetteInput} - Trouvé : {findMaxIndex(output)}")
        tauxIncorrect = nbIncorrect / len(imagesDeTests)
        endTime = time.time()
        duration = (endTime - startTime) * 1000

        print(f"{iteration};{tauxIncorrect};{errorBackPropagate};{duration}" + (f";{mlp.getLearningRate()}" if leanringRateDegressif else ""))

        if tauxIncorrect < tauxSortie:
            learned = True
            allEndTime = time.time()
            print(f"Appris en {(allEndTime - allTime) * 1000} ms")


def intToArray(input):
    result = np.zeros(10)
    result[input] = 1
    return result


def toOneArray(array):
    return array.flatten() / 255.0


def findMaxIndex(array):
    return np.argmax(array)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
