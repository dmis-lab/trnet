from __future__ import unicode_literals, print_function, division
import torch
from torch.autograd import Variable
import torch.nn as nn

class MODEL(nn.Module):
	def __init__(self, 
				 profile_dim,
				 fingerprint_dim,
				 mlp_hidden_states_cell_line,
				 mlp_hidden_states_drug,
				 dropout_rate):
		
		super(MODEL, self).__init__()

		"""
			Model Hyperparameters
		"""
		self.mlp_hidden_states_cell_line = mlp_hidden_states_cell_line
		self.mlp_hidden_states_drug = mlp_hidden_states_drug
		self.dropout_rate = dropout_rate

		"""
			Model Parameters
		"""
		self.profile_dim = profile_dim
		self.fingerprint_dim = fingerprint_dim

		"""
			Graph
		"""
		self.mlp_cell = nn.ModuleList([nn.Sequential(
						nn.Dropout(self.dropout_rate),
						nn.Linear(self.mlp_hidden_states_cell_line[i],
								  self.mlp_hidden_states_cell_line[i+1]),
						nn.ReLU()) for i \
						in xrange(len(self.mlp_hidden_states_cell_line)-2)])
		self.cell_embedding = nn.Sequential(
						nn.Dropout(self.dropout_rate),
						nn.Linear(self.mlp_hidden_states_cell_line[-2],
								  self.mlp_hidden_states_cell_line[-1]))

		mlp_drug = [nn.Sequential(
						nn.Dropout(self.dropout_rate),
						nn.Linear(self.mlp_hidden_states_drug[i],
								  self.mlp_hidden_states_drug[i+1]),
						nn.ReLU()) for i \
						in xrange(len(self.mlp_hidden_states_drug)-2)]
		mlp_drug.append(nn.Sequential(
						nn.Dropout(self.dropout_rate),
						nn.Linear(self.mlp_hidden_states_drug[-2],
								  self.mlp_hidden_states_drug[-1])))
		self.mlp_drug = nn.ModuleList(mlp_drug)

		regression_input = self.mlp_hidden_states_drug[-1] \
		+ self.mlp_hidden_states_cell_line[-1] + 2
		self.lineaer_regression = nn.Sequential(
			nn.Linear(regression_input, int(self.profile_dim/2)),
			nn.ReLU(),
			nn.Linear(int(self.profile_dim /2), self.profile_dim),
			nn.ReLU(),
			nn.Linear(self.profile_dim, self.profile_dim)
			)

	def forward(self, pre_treatments, drugs, doses, times):
		for module in self.mlp_cell:
			pre_treatments = module(pre_treatments)

		cell_embedding = self.cell_embedding(pre_treatments)

		for module in self.mlp_drug:
			drugs = module(drugs)

		prediction = self.lineaer_regression(
			torch.cat([cell_embedding, drugs, 
				doses.unsqueeze(dim=1), times.unsqueeze(dim=1)], dim=1))

		return prediction