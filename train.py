from __future__ import unicode_literals, print_function, division
import sys
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import pickle
import numpy as np
import argparse
import math
from scipy import stats
import utils, parameters, models

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

def train_step(model, criterion, optimizer, data, pre_treatment, 
		 	   drug_fingerprint_info):
	
	model.train()

	step = 0
	loss = .0

	if len(data) % parameters.batch_size == 0:
		batch_num = int(len(data)/parameters.batch_size)
	else:
		batch_num = int(len(data)/parameters.batch_size) + 1

	batches = utils.make_batches(data, pre_treatment, 
								 drug_fingerprint_info,
								 parameters.batch_size)
	for batch in batches:
		pre_treatments, drugs, targets, doses, times = batch
		input_pre_treatments = torch.cuda.FloatTensor(pre_treatments)
		input_drugs = torch.cuda.FloatTensor(drugs)
		input_targets = torch.cuda.FloatTensor(targets)
		input_doses = torch.cuda.FloatTensor(doses)
		input_times = torch.cuda.FloatTensor(times)

		# Optimizing
		optimizer.zero_grad()
		predicted = model(input_pre_treatments, input_drugs, input_doses,
						  input_times)
		_loss = criterion(predicted, input_targets).mean()
		_loss.backward()
		loss+=_loss.item()
		optimizer.step()
		
		step+=1

		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Training Batch: [{}/{}]".format(step, batch_num))                     

	return loss/batch_num

def eval_step(model, criterion, data, cell_line_info, drug_fingerprint_info):
	
	model.eval()

	step = 0
	loss = .0
	l1, l2, CD, ED, pearson, r_squared = .0, .0, .0, .0, .0, .0

	if len(data) % parameters.batch_size == 0:
		batch_num = int(len(data)/parameters.batch_size)
	else:
		batch_num = int(len(data)/parameters.batch_size) + 1

	batches = utils.make_batches(data, cell_line_info, 
								 drug_fingerprint_info,
								 parameters.batch_size, False)
	for batch in batches:
		cell_lines, drugs, targets, doses, times = batch

		with torch.no_grad():
			input_cell_lines = torch.cuda.FloatTensor(cell_lines)
			input_drugs = torch.cuda.FloatTensor(drugs)
			input_targets = torch.cuda.FloatTensor(targets)
			input_doses = torch.cuda.FloatTensor(doses)
			input_times = torch.cuda.FloatTensor(times)

			# Optimizing
			predicted = model(input_cell_lines, input_drugs,  input_doses, 
				input_times)
			l1 += torch.abs(predicted - input_targets).sum()
			l2 += criterion(predicted, input_targets)
			CD += nn.functional.cosine_similarity(predicted, 
																input_targets)\
																.sum()
			ED += torch.sqrt(torch.mul(predicted-input_targets,
				predicted-input_targets).sum(dim=1)).sum()

			for i in xrange(len(input_targets)):
				pearson += stats.pearsonr(input_targets[i], predicted[i])[0]
			r_squared += rsquared(input_targets, predicted).sum()
		
		step+=1

		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Validation Batch: [{}/{}]".format(step, batch_num))                     

	return l1/len(data), l2/len(data), 1 - CD/len(data), ED/len(data), \
	pearson/len(data), r_squared/len(data)

def data_processing(parameters, dataset_dir):
	"""
		Data Processing
	"""
	train, validation, OO, NO, ON, NN, pre_treatment, drug_fingerprint_info \
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

	return train, validation, OO, NO, ON, NN, pre_treatment, \
	drug_fingerprint_info

def load_model():
	model = models.MODEL(parameters.profile_dim, 
						 parameters.fingerprint_dim,
						 parameters.mlp_hidden_states_cell_line,
						 parameters.mlp_hidden_states_drug,
						 parameters.dropout_rate).cuda()

	return model

def main(dataset_dir):

	"""
		Printing Hyperparameters
	"""
	utils.print_hyperparameter(parameters)

	"""
		Dataset Preparation
	"""	
	train, validation, OO, NO, ON, NN, pre_treatment, drug_fingerprint_info \
	= data_processing(parameters, dataset_dir)

	"""
		Model Initializing
	"""
	print("=================================================================")
	print("Model Initializing..")
	model = load_model()

	optimizer = optim.Adam(filter(lambda p: p.requires_grad,
		model.parameters()), lr=parameters.learning_rate)

	criterion = nn.MSELoss(reduction="sum")

	count = 0
	for k in xrange(parameters.epochs):
		"""
			Training Step
		"""
		loss = train_step(model, criterion, optimizer, train, 
		 	   			  pre_treatment, drug_fingerprint_info)

		"""
			Evaluation Step
		"""

		l1, l2, CD, ED, PC, R2 = eval_step(model, criterion, validation, 
	 	   			  pre_treatment, drug_fingerprint_info)
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Epoch: [{}/{}] New Scores: [L1/L2/CD/ED/PC/R2]: [{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}]\n"\
			.format(k+1, parameters.epochs, l1, l2, CD, ED, PC, R2))

		# """
		# 	When you want to save the model, use the following code.
		# """
		# sys.stdout.write("\033[F")
		# sys.stdout.write("\033[K")
		# print("Current Best:)\n")

		# torch.save(model, "best_model_{}_{}.pt".format(i,j))
	
if __name__ == "__main__":
	args = load_argument()
	main(args.dataset_dir)