from rnn.experiments import ExperimentArchitectureRecurrentRegularization, \
	ExperimentArchitectureRecurrentDropout, ExperimentArchitecturesRegularizationActivation
import pandas as pd


dataset = pd.read_pickle(r'..\..\outputs\dataset_train.pickle')
if len(dataset.columns) > 5:
	dataset = dataset.iloc[:, 1:]
dataset.columns = ['x', 'y', 'ekstrakcja', 'klasa', 'indeks']
classes = {'Cienie': 0, 'Drogi': 1,
			'Drzewa iglaste': 2, 'Drzewa lisciaste': 3,
			'Dzewa iglaste': 2, 'Fil_ulm': 4,
			'Mol_cae': 5,  'Pola uprawne': 6,
			'X_niegatunek': 7, 'Zabudowa': 8}
dataset['klasa_id'] = dataset.klasa.map(classes)

ExperimentArchitecturesRegularizationActivation(20, dataset, r'results/rnn_architecture_regularization.csv')
ExperimentArchitectureRecurrentRegularization(20, dataset, r'results/rnn_architecture_regularization_recurrent.csv')
ExperimentArchitectureRecurrentDropout(20, dataset, r'results/rnn_architecture_dropout_recurrent.csv')