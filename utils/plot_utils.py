import os
import matplotlib.pyplot as plt
import time
from constants import dir_predict_actuals
from constants import file_predict_actuals

def plot_predictions_vs_actuals(predictions, actuals, save_dir=dir_predict_actuals, filename=file_predict_actuals):

    plt.rcParams["text.usetex"] = False
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.2, edgecolors='k')
    plt.plot([1, 5], [1, 5], color='red', linestyle='--')

    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Predictions vs. Actual Ratings (1-5 Scale)')
    plt.xlim(1, 5)
    plt.ylim(1, 5)

    filename = filename + "_" + time.strftime("%Y%m%d-%H%M%S") + '.png'
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)
    plt.close()

def plot_training_validation_performance(train_losses, val_losses, n_epochs, save_dir=dir_predict_actuals, filename=file_predict_actuals):
    plt.rcParams["text.usetex"] = False
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(save_dir, filename + '_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
    plt.savefig(file_path)
    plt.close()
