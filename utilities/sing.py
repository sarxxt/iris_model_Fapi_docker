import pickle

class SingletonLogger():
    _instance = None

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            cls._instance = pickle.load(open("utilities/clf_model.pkl", "rb"))
            print("check")
        return cls._instance    
            

