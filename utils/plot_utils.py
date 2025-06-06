import os
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from constants import dir_predict_actuals, file_predict_actuals, dir_training_validation_performance, file_training_validation_performance

def plot_predictions_vs_actuals(predictions, actuals, save_dir=dir_predict_actuals, filename=file_predict_actuals):
    plt.rcParams["text.usetex"] = False
    os.makedirs(save_dir, exist_ok=True)

    # Combine predictions and actuals into a DataFrame
    import pandas as pd
    data = pd.DataFrame({'Actual Ratings': actuals, 'Predicted Ratings': predictions})

    # Create the violin plot
    plt.figure(figsize=(8, 8))
    sns.violinplot(x='Actual Ratings', y='Predicted Ratings', data=data, density_norm='width', inner='quartile')

    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Violin Plot: Predictions vs. Actual Ratings (1-5 Scale)')
    plt.ylim(1, 5)

    # Save the plot
    filename = filename + "_" + time.strftime("%Y%m%d-%H%M%S") + '.png'
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)
    plt.close()

def plot_training_validation_performance(train_losses, val_losses, n_epochs, save_dir=dir_training_validation_performance, filename=file_training_validation_performance):
    plt.rcParams["text.usetex"] = False
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(save_dir, filename + '_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
    plt.savefig(file_path)
    plt.close()
