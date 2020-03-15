from config import Config
from rnn.experiments import ExperimentArchitecture
import pandas as pd


dataset = pd.read_pickle(Config.dataset_train)
if len(dataset.columns) > 5:
    dataset = dataset.iloc[:, 1:]
dataset.columns = ["x", "y", "ekstrakcja", "klasa", "indeks"]
classes = {
    "Cienie": 0,
    "Drogi": 1,
    "Drzewa iglaste": 2,
    "Drzewa lisciaste": 3,
    "Dzewa iglaste": 2,
    "Fil_ulm": 4,
    "Mol_cae": 5,
    "Pola uprawne": 6,
    "X_niegatunek": 7,
    "Zabudowa": 8,
}
dataset["klasa_id"] = dataset.klasa.map(classes)

ExperimentArchitecture(20, dataset, r"results/rnn_architecture.csv")
