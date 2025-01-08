# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.backends.cudnn
from scipy.stats import norm

__all__ = ['draw_loss',
           'time_counter',
           'PDF_Plotter',
           'f_max',
           'seed_everything',
           'EarlyStopping',
           'Prediction_plot',
           'Folder_checker',
           'time_performance',
           'finite_difference',
           'plot_histogram']


def plot_histogram(R_list):
    """
    # Plot histogram
    plot_histogram(R_list, 'Correlation Coefficient', 'Frequency', 'Histogram of Correlation Coefficients')
    """
    plt.figure(figsize=(4.8, 5.2), frameon=True)
    plt.rcParams['savefig.dpi'] = 600  # Image resolution

    # Create histogram
    plt.hist(R_list, bins='auto', edgecolor='black',
             density=False)  # Set density parameter to True
    plt.grid(axis='y', alpha=0.75)

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Model Coefficients')

    # Reverse x-axis
    plt.gca().invert_xaxis()
    # plt.savefig('result.tif')
    # Display histogram
    plt.show()


class finite_difference:
    def __init__(self, data_len, dt):
        self.data_len = data_len
        self.dt = dt
        self.phi = self.finite_difference1()

    def finite_difference1(self):
        phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]),
                               np.zeros([self.data_len - 3, ])], dtype=np.float64)
        temp1 = np.concatenate([-1 / 2 * np.identity(self.data_len - 2),
                                np.zeros([self.data_len - 2, 2])], axis=1, dtype=np.float64)
        temp2 = np.concatenate([np.zeros([self.data_len - 2, 2]),
                                1 / 2 * np.identity(self.data_len - 2)], axis=1, dtype=np.float64)
        phi2 = temp1 + temp2
        phi3 = np.concatenate([np.zeros([self.data_len - 3, ]), np.array([1 / 2, -2, 3 / 2])], dtype=np.float64)
        Phi_t = 1 / self.dt * np.concatenate(
            [np.reshape(phi1, [1, phi1.shape[0]]),
             phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0, dtype=np.float64)
        Phi_t = torch.tensor(Phi_t, dtype=torch.float64)
        return Phi_t

    def finite_difference2(self, input_data):
        input_data = input_data.to(torch.float64)
        Phi_t = self.phi.to(input_data.device).to(torch.float64)
        # Convert phi to match input_data1's data type and device
        if len(input_data.shape) == 3:
            # If input_data1 is 3D, perform the following operation
            dot = torch.matmul(Phi_t, input_data[:, :, 0].permute(1, 0)).permute(1, 0)
        elif len(input_data.shape) == 2:
            # If input_data1 is 2D, perform the following operation
            dot = torch.matmul(Phi_t, input_data.permute(1, 0)).permute(1, 0)
        else:
            # If input_data1 is neither 2D nor 3D, raise an exception
            raise ValueError("Input data must be either 2D or 3D.")
        return dot[:, :, None].to(torch.float32)


def Prediction_plot(Target_true, Target_pred, Title='title',
                    Is_plot_show=False):
    for Sample in range(len(Target_true)):
        if Is_plot_show:
            plt.figure()
            plt.plot(Target_true[Sample, :], label='True')
            plt.plot(Target_pred[Sample, :], label='Predict')
            plt.title(Title)
            plt.legend()
            plt.show()
            plt.close()


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def Folder_checker(path_tuple):
    for path in path_tuple:
        if not os.path.exists(path):
            os.makedirs(path)


def draw_loss(picture_path: str,
              train_loss, val_loss, plt_save=False, plt_show=False):  # Loss plotting function
    plt.figure(figsize=(8, 6), frameon=True)
    plt.rcParams['font.family'] = 'MicroSoft YaHei'  # Set font for character display
    plt.plot(np.log(train_loss), label='train_loss')
    plt.plot(np.log(val_loss), label='val_loss')
    plt.title('Loss Function Plot')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Value')
    plt.grid(visible=1, which='major', axis='both', linestyle='--')  # Display grid lines
    plt.legend(loc='upper right')
    if plt_save:
        if not os.path.exists(picture_path + 'loss/'):  # Create directory if it doesn't exist
            os.makedirs(picture_path + 'loss/')
        plt.savefig(picture_path + 'loss/loss.png')
    if plt_show:
        plt.show()
    plt.close()


def time_counter(begin_time, end_time):
    """
        Time counter: Calculate training duration and convert timestamp to hours, minutes, and seconds
        Running time is rounded using run_time.round() function
    """
    run_time = round(end_time - begin_time)
    # Calculate hours, minutes, seconds
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    # Output
    print(f'Training time: {hour} hours {minute} minutes {second} seconds')


def PDF_Plotter(data_name: str, picture_path: str,
                true_values, predicted_values, threshold=0.05, plt_save=False,
                plt_show=True):
    # Convert data to float32
    true_values = np.array(true_values).astype(np.float32)
    predicted_values = np.array(predicted_values).astype(np.float32)

    # Calculate differences
    differences = predicted_values - true_values

    # Calculate maximum differences along each row
    max_differences = np.max(np.abs(true_values), axis=1)

    # Calculate normalized differences
    normalized_differences = differences / max_differences[:, np.newaxis]

    # Flatten the normalized differences
    total_differences = normalized_differences.flatten()

    # Fit a normal distribution to the data
    '''from scipy.stats import norm'''
    mu, sigma = norm.fit(total_differences)
    pdf = norm.pdf(total_differences, mu, sigma)

    # Calculate the number of values within the threshold range
    num_values_within_threshold = np.sum(np.logical_and(total_differences > -threshold, total_differences < threshold))
    pdf_metric = num_values_within_threshold / len(total_differences)

    print(data_name + f"Probability: {pdf_metric * 100:.2f}%")

    plt.figure(figsize=(7.2, 5.2), frameon=True)
    plt.rcParams['savefig.dpi'] = 600  # Image resolution
    plt.rcParams['font.family'] = 'MicroSoft YaHei'  # Set font for Chinese character support
    plt.rcParams['axes.linewidth'] = 2.5  # Set axis line width
    plt.scatter(total_differences, pdf, s=3, alpha=0.5, color='black', marker=".")
    plt.title('Probability Density Function')
    plt.xlabel('Normalized Differences')
    plt.ylabel('Probability Density')

    # Add two vertical lines
    plt.axvline(x=-threshold, color='blue', linestyle='dashdot', linewidth=2.5, label='Threshold')
    plt.axvline(x=threshold, color='blue', linestyle='dashdot', linewidth=2.5)

    # Set ticks to display inside the frame
    plt.tick_params(axis='both', direction='in', length=6, width=1, colors='black')

    # Adjust plot boundaries
    value_max = max(abs(np.max(total_differences)), abs(np.min(total_differences)))
    plt.xlim(-value_max, value_max)

    if plt_save:
        if not os.path.exists(picture_path + 'PDF/'):  # Create directory if it doesn't exist
            os.makedirs(picture_path + 'PDF/')
        plt.savefig(picture_path + 'PDF/PDF figure.png')
    if plt_show:
        plt.show()
    plt.close()

    return pdf_metric


def f_max(y_true, y_pred):
    """
    Calculate maximum peak error and average peak error
    :param y_true: True values
    :param y_pred: Predicted values
    :return: Maximum peak error and average peak error
    """
    # Check input parameter types
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Input parameters must be NumPy arrays")

    # Check input parameter shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("True values and predicted values must have the same shape")

    max_true = np.abs(y_true).max(axis=1)  # Calculate maximum absolute value of true values
    max_pred = np.abs(y_pred).max(axis=1)  # Calculate maximum absolute value of predicted values

    dif_ = np.abs(max_true - max_pred)  # Calculate absolute error
    max_dif_rate_list = dif_ / max_true * 100  # Calculate error percentage list
    average_dif_rate = np.mean(max_dif_rate_list)  # Calculate average error percentage
    max_dif_rate = np.max(max_dif_rate_list)  # Calculate maximum error percentage

    print('AvgPE: %.3f%%' % average_dif_rate, '   MaxPE: %.3f%%' % max_dif_rate)

    return max_dif_rate, average_dif_rate


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=True, delta=0, path='../Checkpoint/' + '_bestmodel.pt', trace_func=print):
        """

        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}'
                            f', the best performance is: {-self.best_score:.3e}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.3e} --> {val_loss:.3e}).  Saving model ...')
            torch.save(model, self.path, pickle_protocol=3)

        self.val_loss_min = val_loss


def time_performance(label, pred):
    MSE = np.mean((pred - label) ** 2)
    std_label = np.sqrt(np.mean((label - np.mean(label, axis=1)[:, None]) ** 2, axis=1))
    RMSE = np.mean(np.sqrt(np.mean((pred - label) ** 2, axis=1)) / std_label)
    MAE = np.mean(np.abs(pred - label))
    RMAE = np.mean(np.max(np.abs(pred - label), axis=1) / std_label)
    r_all = []
    for i in range(label.shape[0]):
        r_all.append(np.corrcoef(label[i, :].ravel(), pred[i, :].ravel())[0, 1])
    r = np.mean(r_all)
    print('r: %.1f' % (100 * r))
    return MSE, RMSE, MAE, RMAE, r, r_all
