# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from basecode.Loss_function import Loss_PhyHysLSTM
from basecode.NET_model import PhyLSTM3
from basecode.Tools import time_counter, draw_loss, PDF_Plotter, Prediction_plot, f_max, \
    time_performance, seed_everything, Folder_checker, finite_difference, EarlyStopping, plot_histogram

warnings.filterwarnings("ignore", message="np.find_common_type is deprecated.*")


class Dataset:
    def __init__(self, datafile, slice_start, slice_end, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.mat = sio.loadmat(datafile)
        self.X = torch.as_tensor(self.mat['input'][slice_start:slice_end])
        self.Y_u = torch.as_tensor(self.mat['target_u'][slice_start:slice_end])
        self.Y_v = torch.as_tensor(self.mat['target_v'][slice_start:slice_end])
        self.Y_a = torch.as_tensor(self.mat['target_a'][slice_start:slice_end])
        self.Y_g = torch.as_tensor(self.mat['target_g'][slice_start:slice_end])
        self.Xc = torch.as_tensor(self.mat['input'][0:50]).float().cuda()
        self.Y = torch.stack([self.Y_u, self.Y_v, self.Y_a, self.Y_g], dim=2)
        self.process_data()

    def process_data(self):
        if len(self.X.shape) == 2:
            self.X = self.X.unsqueeze(2)
        if len(self.Xc.shape) == 2:
            self.Xc = self.Xc.unsqueeze(2)
        if len(self.Y.shape) == 2:
            self.Y = self.Y.unsqueeze(2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_data = self.X[idx]
        target_data = self.Y[idx]

        if self.transform:
            input_data = self.transform(input_data)

        if self.target_transform:
            target_data = self.target_transform(target_data)

        return (input_data, self.Xc), target_data


def train_one_epoch1(scaler, Config):
    Config.model_train1.train()
    train_loss = 0
    for batch, (ag_train, target_train) in enumerate(Config.trainLoader1):
        batch_ag_train = Variable(ag_train[0])
        batch_ag_c_train = Variable(ag_train[1][0])
        batch_target_train = Variable(target_train)

        batch_pred_train = Config.model_train1(batch_ag_train, model_type='data')
        batch_pred_c_train = Config.model_train1(batch_ag_c_train, model_type='phy')
        batch_train_loss = Config.loss_fn1((batch_ag_train, batch_ag_c_train),
                                           (batch_pred_train, batch_pred_c_train),
                                           batch_target_train)
        Config.optimizer1.zero_grad()
        train_loss += batch_train_loss.item() / len(ag_train)
        scaler.scale(batch_train_loss).backward()
        scaler.step(Config.optimizer1)
        scaler.update()
    return train_loss


def val_one_epoch1(Config):
    Config.model_train1.eval()
    val_loss = 0
    with torch.no_grad():
        for batch, (ag_val, batch_target_val) in enumerate(
                Config.trainLoader1):
            batch_input_val = ag_val[0]
            batch_input_c_val = ag_val[1][0]
            batch_pred_val = Config.model_train1(batch_input_val, model_type='data')
            batch_pred_c_val = Config.model_train1(batch_input_c_val, model_type='phy')
            batch_val_loss = Config.loss_fn1(
                (batch_input_val.cpu(), batch_input_c_val.cpu()),
                (batch_pred_val.cpu(), batch_pred_c_val.cpu()),
                batch_target_val.cpu())
            val_loss += batch_val_loss.item() / len(ag_val)
    return val_loss


def train_loop1(Config):
    Train_loss1 = []
    Val_loss1 = []
    start_time_all = time.time()
    scaler = GradScaler()
    for epoch in range(Config.max_epoch1):
        Config.model_train1.train()
        start_time = time.time()
        train_loss = train_one_epoch1(scaler, Config)
        # val_loss = val_one_epoch1(Config)
        val_loss = train_loss
        Train_loss1.append(train_loss)
        Val_loss1.append(val_loss)

        Config.early_stopping1(val_loss, Config.model_train1)
        if Config.early_stopping1.early_stop:
            print('Epoch: %d, Train loss: %.5e, Val loss: %.5e'
                  % (epoch, train_loss, val_loss))
            break
        if epoch % 1 == 0:
            time_length = time.time() - start_time
            print('Epoch: %d, Train loss: %.5e, Val loss: %.5e, Time: %.2f秒'
                  % (epoch, train_loss, val_loss, time_length))

    time_counter(start_time_all, time.time())
    torch.save(
        Config.model_train1,
        Config.model_path1 + 'FModel.pth',
        pickle_protocol=3)
    draw_loss(
        picture_path=Config.picture_path1,
        train_loss=Train_loss1,
        val_loss=Val_loss1,
        plt_save=Config.plt_save,
        plt_show=Config.plt_show)


def test_loop1(Config, model_test):
    model_test.eval()
    test_input = []
    test_target = []
    test_pred = []

    with torch.no_grad():
        for batch, (batch_input_test, batch_target_test) in enumerate(Config.testLoader1):
            batch_input_test = batch_input_test[0]
            batch_pred_test = model_test(batch_input_test, model_type='data')
            test_input.append(batch_input_test.cpu().squeeze().numpy())
            test_target.append(batch_target_test.cpu().squeeze().numpy())
            test_pred.append(batch_pred_test.cpu().squeeze().numpy())

    return np.array(test_input)[0], np.array(test_pred)[0], np.array(test_target)[0]


def test_loop2(Config, model_test):
    model_test.eval()
    test_input = []
    test_target = []
    test_pred = []
    Net1_model = torch.load(Config.testmodel_path1)
    Net1_model.eval()

    with torch.no_grad():
        for batch, (batch_input_test, batch_target_test) in enumerate(Config.testLoader2):
            batch_pred_Z_test = Net1_model(batch_input_test[0], model_type='data')
            batch_pred_test = model_test(batch_pred_Z_test)
            test_input.append(batch_input_test[0].cpu().squeeze().numpy())
            test_target.append(batch_target_test.cpu().squeeze().numpy())
            test_pred.append(batch_pred_test.cpu().squeeze().numpy())

    return np.array(test_input)[0], np.array(test_pred)[0], np.array(test_target)[0]


def Test_And_Save_Processing(Config):
    model_test = torch.load(Config.testmodel_path1)
    test_input, test_pred, test_target = test_loop1(Config=Config, model_test=model_test)
    test_pred, test_target = test_pred[:, :, 0:1], test_target[:, :, 0:1]
    # test_pred, test_target = test_pred[:, :, 1:2], test_target[:, :, 1:2]
    # test_pred, test_target = test_pred[:, :, 2:3], test_target[:, :, 2:3]
    # test_pred, test_target = test_pred[:, :, 3:4], test_target[:, :, 3:4]
    CI99_PDF = PDF_Plotter(
        data_name='CI98%',
        picture_path=Config.picture_path1,
        true_values=test_target,
        predicted_values=test_pred,
        threshold=0.02,
        plt_save=True,
        plt_show=False)

    Prediction_plot(
        Target_true=test_target,
        Target_pred=test_pred,
        Title='a_Test',
        Is_plot_show=Config.plt_show)

    PeakError_max, PeakError_avg = f_max(y_true=test_target, y_pred=test_pred)
    MSE_t, RMSE_t, MAE_t, RMAE_t, r_t, R_list = time_performance(label=test_target, pred=test_pred)
    plot_histogram(R_list)

    sio.savemat(file_name=Config.result_path1 + 'result.mat',
                mdict={'test_input': test_input,
                       'test_pred': test_pred,
                       'test_target': test_target})


class ModelConfig:
    def __init__(self, data_name):
        self.data_name = data_name  # "pure_knetIBRH13"
        self.target_rate = 1
        self.train_num = 10
        self.val_num = 0
        self.test_num = 90
        self.train_batch = 10  # Batch size of training data
        self.dt = 0.02

        self.model_type1 = 'baseline_1'
        self.learning_rate = 1e-3
        self.max_epoch1 = 20000
        self.N_feature = 1
        self.N_hidden = 100
        self.N_target = 1
        self.data_len = 1501
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.plt_save = True
        self.plt_show = False
        self.need_val = False
        self.BestModel = False

        self.filename1 = '%s_%s_hidden%d_batch%d_lr%0.1e_epoch%d_TgRate%d' % (
            self.model_type1, self.data_name, self.N_hidden, self.train_batch,
            self.learning_rate, self.max_epoch1, self.target_rate)
        self.data_path = "../data/"
        self.model_path1 = f"results/{self.filename1}/model_save/"
        self.result_path1 = f"results/{self.filename1}/results/"
        self.picture_path1 = f"results/{self.filename1}/pictures/"
        self.testmodel_path1 = self.model_path1 + 'BModel.pth' if self.BestModel else self.model_path1 + 'FModel.pth'

        Folder_checker(path_tuple=(self.data_path, self.model_path1, self.result_path1, self.picture_path1))

        self.finite_difference = finite_difference(data_len=self.data_len, dt=self.dt)
        self.trainLoader1, self.testLoader1 = self.get_data_loader(is_run1=True,
                                                                   need_val=self.need_val)
        self.early_stopping1 = EarlyStopping(
            patience=1000, verbose=True, path=self.model_path1 + 'BModel.pth')
        self.loss_fn1 = Loss_PhyHysLSTM(is_run1=True, Dif=self.finite_difference)
        self.model_train1 = PhyLSTM3(n_hidden=self.N_hidden,
                                     Fin_DIF=self.finite_difference).to(self.device)
        self.optimizer1 = torch.optim.Adam(self.model_train1.parameters(), self.learning_rate)

    def get_transform(self):
        return Lambda(lambda x_: torch.as_tensor(x_ * self.target_rate).float().cuda())

    def get_data_loader(self, is_run1=True, need_val=True):
        def create_data_slice(start, end, batch_size, need_shuffle=True):
            data = Dataset(datafile=self.data_path + self.data_name + ".mat", transform=self.get_transform(),
                           slice_start=start, slice_end=end, target_transform=self.get_transform())
            return DataLoader(data, batch_size=batch_size, shuffle=need_shuffle)

        test_loader = create_data_slice(start=self.train_num + self.val_num,
                                        end=self.train_num + self.val_num + self.test_num,
                                        batch_size=self.test_num, need_shuffle=False)
        if need_val:
            train_loader = create_data_slice(
                start=0, end=self.train_num, batch_size=self.train_batch)
            val_loader = create_data_slice(start=self.train_num,
                                           end=self.train_num + self.val_num,
                                           batch_size=self.val_num, need_shuffle=False)

            return train_loader, val_loader, test_loader
        else:
            train_loader = create_data_slice(start=0, end=self.train_num + self.val_num,
                                             batch_size=self.train_batch)
            return train_loader, test_loader


def main():
    # train_loop1(config)
    Test_And_Save_Processing(Config=config)


if __name__ == '__main__':
    seed_everything(1234)
    dataname = "BLSdof"
    config = ModelConfig(data_name=dataname)
    print(config.data_name, config.filename1)
    main()
