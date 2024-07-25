import pandas as pd

class Clean():

    def __init__(self, file):
        self.file = file

    def preprocess(self):
         self.df = pd.read_csv(self.file)
         self.df.rename(columns={'SepalLengthCm': 'sepal_length', 'SepalWidthCm': 'sepal_width', 'PetalLengthCm': 'petal_length', 'PetalWidthCm': 'petal_width', 'Species': 'species'}, inplace=True)
         self.df = self.df.drop(['species','Id'], axis =1)
         return self.df
    