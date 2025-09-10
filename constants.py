n_epochs = 50 # todo: try 50 - 100.
batch_size = 256
lr = 0.001
forward_ratio = 0.8
seed = 42


dir_predict_actuals = "plots_predict_actual/"
file_predict_actuals = "predict_actuals"

dir_training_validation_performance = "plots_training_validation_performance/"
file_training_validation_performance = "training_validation_performance"

TOP_N_RECOMMENDATIONS = 10

# Multi-objective training hyperparameters
lambda_rmse = 0.7    # Weight for RMSE loss
lambda_novelty = 0.15  # Weight for novelty loss
lambda_diversity = 0.15  # Weight for diversity loss