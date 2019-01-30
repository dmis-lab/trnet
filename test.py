from __future__ import unicode_literals, print_function, division
import sys
import utils, parameters, models
import torch.optim as optim
import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import time
import argparse
import pickle
import numpy as np
from scipy import stats

def load_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_dir', type=str, 
						default="./TRNet_Dataset/", help='Dataset Directory')
	args = parser.parse_args()

	return args

def rsquared(x, y):
	avg = torch.mean(y, dim=1, keepdim=True)

	top = torch.sum((x-y)**2, dim=1)
	bottom = torch.sum((y - avg)**2, dim=1)

	return 1 - top/bottom

def eval_step(models, criterion, data, cell_line_info, drug_fingerprint_info):
	
	for model in models:
		model.eval()

	step = 0

	MAE, MSE, CD, ED, PC, R2 = .0, .0, .0, .0, .0, .0

	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("Evaluation..")


	for line in data:
		cell_line, drug, target, dosage, time = line

		with torch.no_grad():
			input_pre_treatments = torch.cuda\
			.FloatTensor([cell_line_info[cell_line]])
			input_drugs = torch.cuda\
			.FloatTensor([drug_fingerprint_info[drug][0]])
			input_targets = torch.cuda.FloatTensor([target])
			input_doses = torch.cuda.FloatTensor([dosage])
			input_times = torch.cuda.FloatTensor([time])

			# Optimizing
			predicted = [model(input_pre_treatments, input_drugs,  input_doses,\
				input_times) for model in models]
			predicted = sum(predicted)/len(predicted)

			MAE += torch.abs(input_targets - predicted).sum()
			MSE += criterion(input_targets, predicted)
			CD += 1 - nn.functional.cosine_similarity(input_targets, predicted)[0]
			ED += torch.sqrt(torch.mul(input_targets - predicted, input_targets - predicted).sum())
			PC += stats.pearsonr(input_targets[0], predicted[0])[0]
			R2 += rsquared(predicted, input_targets).sum()

			step+=1

			sys.stdout.write("\033[F")
			sys.stdout.write("\033[K")
			print("Evaluation [{}/{}]".format(step, len(data)))
	return MAE/len(data), MSE/len(data), CD/len(data), ED/len(data),\
	 PC/len(data), R2/len(data)

def data_processing(dataset_dir):
	"""
		Data Processing
	"""
	train, validation, OO, NO, ON, NN, \
	pre_treatment, drug_fingerprint_info \
	= utils.data_processing(dataset_dir,
							parameters.train_data_path,
							parameters.validation_data_path,
							parameters.OO_data_path,
							parameters.NO_data_path,
							parameters.ON_data_path,
							parameters.NN_data_path,
							parameters.pre_treatment_data_path,
							parameters.drug_fingerprint_data_path
							)

	"""
		Printing Data Statistics
	"""
	utils.print_statistics(len(train),
						   len(validation),
						   len(OO),
						   len(NO),
						   len(ON),
						   len(NN),
						   len(pre_treatment),
						   len(drug_fingerprint_info)
						   )

	return train, validation, OO, NO, ON, NN, \
	pre_treatment, drug_fingerprint_info

def main(dataset_dir):

	"""
		Printing Hyperparameters
	"""
	utils.print_hyperparameter(parameters)

	"""
		Dataset Preparation
	"""	
	train, validation, OO, NO, ON, NN, pre_treatment, drug_fingerprint_info \
	= data_processing(dataset_dir)

	"""
		Model Initializing (using Ensemble)
	"""
	print("=================================================================")
	print("Model Initializing..")
	models = [torch.load("./best_models/best_model_3_0.pt"),
			  torch.load("./best_models/best_model_6_1.pt"),
			  torch.load("./best_models/best_model_0_0.pt"),
			  torch.load("./best_models/best_model_3_1.pt"),
  			  torch.load("./best_models/best_model_6_2.pt"),
  			  torch.load("./best_models/best_model_0_2.pt"),
  			  torch.load("./best_models/best_model_1_2.pt"),
  			  torch.load("./best_models/best_model_7_0.pt"),
  			  torch.load("./best_models/best_model_4_2.pt"),
  			  torch.load("./best_models/best_model_7_2.pt")]
			  			  
	criterion = nn.MSELoss(reduction="sum")

	"""
		Initial Scores Before Training
	"""

	MAE, MSE, CD, ED, PC, R2 = eval_step(models, criterion, OO, 
	 	   			  pre_treatment, drug_fingerprint_info)
	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("OO: [MAE/MSE/CS/ED/PC/R2]: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\n"\
		.format(MAE, MSE, CD, ED, PC, R2))

	MAE, MSE, CD, ED, PC, R2 = eval_step(models, criterion, NO, 
	 	   			  pre_treatment, drug_fingerprint_info)
	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("NO: [MAE/MSE/CS/ED/PC/R2]: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\n"\
		.format(MAE, MSE, CD, ED, PC, R2))

	MAE, MSE, CD, ED, PC, R2 = eval_step(models, criterion, OO, 
	 	   			  pre_treatment, drug_fingerprint_info)
	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("ON: [MAE/MSE/CS/ED/PC/R2]: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\n"\
		.format(MAE, MSE, CD, ED, PC, R2))

	MAE, MSE, CD, ED, PC, R2 = eval_step(models, criterion, OO, 
	 	   			  pre_treatment, drug_fingerprint_info)
	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("NN: [MAE/MSE/CS/ED/PC/R2]: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\n"\
		.format(MAE, MSE, CD, ED, PC, R2))


if __name__ == "__main__":
	args = load_argument()
	main(args.dataset_dir)
