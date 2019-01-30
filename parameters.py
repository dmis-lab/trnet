"""
	Data Path
"""
train_data_path = "train.pkl"
validation_data_path = "validation.pkl"
OO_data_path = "OO.pkl"
ON_data_path = "ON.pkl"
NO_data_path = "NO.pkl"
NN_data_path = "NN.pkl"
pre_treatment_data_path = "pre_treatment.pkl"
drug_fingerprint_data_path = "drugs_fingerprint.pkl"

"""
	Data Parameter
"""
profile_dim = 978
fingerprint_dim = 2048

"""
	Training Hyperparamenters
"""
epochs = 100
batch_size = 256
learning_rate = 0.001
dropout_rate = 0.2

"""
	Model Hyperparameters
"""
mlp_hidden_states_cell_line = [profile_dim, 800, 400]

mlp_hidden_states_drug = [fingerprint_dim, 600, 200]