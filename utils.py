from __future__ import unicode_literals, print_function, division

import pickle
import random
import numpy as np

def print_hyperparameter(parameters):
	print("=================================================================")
	print("HYPER-PARAMETERS\n")
	print("PROFILE_DIM: {}".format(parameters.profile_dim))
	print("FINGERPRINT_DIM: {}".format(parameters.fingerprint_dim))
	print("EPOCHS: {}".format(parameters.epochs))
	print("BATCH_SIZE: {}".format(parameters.batch_size))
	print("LEARNING_RATE: {}".format(parameters.learning_rate))
	print("DROPOUT_RATE: {}".format(parameters.dropout_rate))
	print("MLP_HIDDEN_CELL_LINE: {}"\
		.format(parameters.mlp_hidden_states_cell_line))
	print("MLP_HIDDEN_DRUG: {}"\
		.format(parameters.mlp_hidden_states_drug))


def print_statistics(train_data_num, validation_data_num, test_random_data_num,
					 test_drug_data_num, test_cell_data_num, test_new_data_num,
					 cell_line_num, drug_num):
	print("=================================================================")
	print("Data Statistics\n")
	print("Number of Train Data: {}".format(train_data_num))
	print("Number of Validation Data: {}".format(validation_data_num))
	print("Number of Test Random Data: {}".format(test_random_data_num))
	print("Number of Test Drug Data: {}".format(test_drug_data_num))
	print("Number of Test Cell Data: {}".format(test_cell_data_num))
	print("Number of Test New Data: {}".format(test_new_data_num))
	print("Number of Cell Line: {}".format(cell_line_num))
	print("Number of Drug: {}".format(drug_num))

def _data_load(dataset_dir, train_data_path, validation_data_path,
			   OO_data_path, NO_data_path, ON_data_path, NN_data_path,
			   pre_treatment_data_path, drug_fingerprint_data_path):

	"""
		Dataset (Train, Validation, Test) Format:
			line[0]: (cell_line,
					  drug, 
					  drug_type, 
					  does, 
					  does_type, 
					  time, 
					  time_type)
			line[1]: 978-dimensional Vector (Post_treatement_profile)

		Cell_line Data Format:
			key: cell_line
			value: 978-dimensional Vector

		Drug Fingerprint Data Format:
			key: drug
			value: 2048-dimenstional Vector
	"""
	print("=================================================================")
	print("Data Loading..")

	with open(dataset_dir+train_data_path, "rb") as f:
		train = pickle.load(f)

	with open(dataset_dir+validation_data_path, "rb") as f:
		validation = pickle.load(f)

	with open(dataset_dir+OO_data_path, "rb") as f:
		OO = pickle.load(f)

	with open(dataset_dir+NO_data_path, "rb") as f:
		NO = pickle.load(f)

	with open(dataset_dir+ON_data_path, "rb") as f:
		ON = pickle.load(f)

	with open(dataset_dir+NN_data_path, "rb") as f:
		NN = pickle.load(f)

	with open(dataset_dir+pre_treatment_data_path, "rb") as f:
		pre_treatment = pickle.load(f)

	with open(dataset_dir+drug_fingerprint_data_path, "rb") as f:
		drug_fingerprint_info = pickle.load(f)

	return train, validation, OO, NO, ON, NN, pre_treatment, \
	drug_fingerprint_info

def _make_input(_train, _validation, _OO, _NO, _ON, _NN):
	"""
		Train, Validation, Test: (cell_line, drug, drug_response_vector)
	"""

	print("Making Input..")

	train = [(line[0][0], line[0][1], line[1], float(line[0][3]), 
		float(line[0][5])) for line in _train]
	validation = [(line[0][0], line[0][1], line[1], float(line[0][3]), 
		float(line[0][5])) for line in _validation]
	OO = [(line[0][0], line[0][1], line[1], float(line[0][3]), 
		float(line[0][5])) for line in _OO]

	NO = [(line[0][0], line[0][1], line[1], float(line[0][3]), 
		float(line[0][5])) for line in _NO]

	ON = [(line[0][0], line[0][1], line[1], float(line[0][3]), 
		float(line[0][5])) for line in _ON]

	NN = [(line[0][0], line[0][1], line[1], float(line[0][3]), 
		float(line[0][5])) for line in _NN]

	return train, validation, OO, NO, ON, NN

def data_processing(dataset_dir, train_data_path, validation_data_path, 
					OO_data_path, NO_data_path, ON_data_path, NN_data_path,
					pre_treatment_data_path, drug_fingerprint_data_path):

	"""
		Data Loading
	"""
	train, validation, OO, NO, ON, NN, pre_treatment, drug_fingerprint_info \
	= _data_load(dataset_dir, 
				 train_data_path,
				 validation_data_path, 
				 OO_data_path,
				 NO_data_path,
				 ON_data_path,
				 NN_data_path,
				 pre_treatment_data_path,
				 drug_fingerprint_data_path)

	"""
		Making Input
	"""
	train, validation, OO, NO, ON, NN = _make_input(train,
													validation,
													OO,
													NO,
													ON,
													NN)

	return train, validation, OO, NO, ON, NN, pre_treatment, \
	drug_fingerprint_info

def make_batches(data, cell_line_info, drug_fingerprint_info,
				 batch_size, is_train=True):
	if is_train:
		random.shuffle(data)

	if len(data) % batch_size == 0:
		batch_num = int(len(data)/batch_size)
	else:
		batch_num = int(len(data)/batch_size) + 1

	for i in xrange(batch_num):
		pre_treatments = []
		drugs = []
		targets = []
		doses = []
		times = []

		left = i*batch_size
		right = min((i+1)*batch_size, len(data))

		for line in data[left:right]:
			pre_treatments.append(cell_line_info[line[0]])
			drugs.append(drug_fingerprint_info[line[1]][0])
			targets.append(line[2])
			doses.append(line[3])
			times.append(line[4])

		yield pre_treatments, drugs, targets, doses, times