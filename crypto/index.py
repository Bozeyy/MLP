import random
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# Définir la classe Index
class Index:
    def __init__(self, start, end, open_, high, low, close, volume, market_cap):
        self.start = start
        self.end = end
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.market_cap = market_cap

    def __repr__(self):
        return (f"Index(start={self.start}, end={self.end}, open={self.open}, high={self.high}, "
                f"low={self.low}, close={self.close}, volume={self.volume}, market_cap={self.market_cap})")

class tabIndex:
    def __init__(self):
        self.index_list = []

    def addIndex(self, index):
        self.index_list.append(index)

    def readFile(self, filename):
        df = pd.read_csv(filename)
        # Itérer sur les lignes du DataFrame à l'envers
        for _, row in df.iloc[::-1].iterrows():
            self.addIndex(row['Close'])
        #     index = Index(
        #         start=row['Start'],
        #         end=row['End'],
        #         open_=row['Open'],
        #         high=row['High'],
        #         low=row['Low'],
        #         close=row['Close'],
        #         volume=row['Volume'],
        #         market_cap=row['Market Cap']
        #     )
        #     self.addIndex(index)

    # méthode getRandomSequence renvoie une séquende de 100 éléments en commençant à un index aléatoire
    def getRandomSequence(self, sequence_length):
        start_index = random.randint(0, len(self.index_list) - sequence_length)
        return self.index_list[start_index:start_index + sequence_length]
