import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
from model import Mamba, MambaConfig
import joblib

LR = 0.001
BATCH_SIZE = 256
TBATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 200

New_data = False

def MambaTrain(FeatureData, labels, epoch, name):
    model_path = '../model/' + name + '/'
    if New_data:
        X_train, X_val, y_train, y_val = train_test_split("random", FeatureData, labels, test_size=0.1, random_state=123)
        X_train, X_test, y_train, y_test = train_test_split("random", X_train, y_train, test_size=1/9, random_state=123)

    else:
        path = '../data/processed/' + name + '/'
        train_data = np.load(path + 'train_df.npy')
        test_data = np.load(path + 'test_df.npy')
        val_data = np.load(path + 'val_df.npy')
        properties_dict = {
            'PH(CaCl2)': -9,
            'PH(H2O)': -8,
            'OC': -6,
            'CaCO3': -5,
            'N': -3
        }
        p = properties_dict[name]
        X_train = train_data[:, 2:1026]
        y_train = train_data[:, p:p+1]
        X_test = test_data[:, 2:1026]
        y_test = test_data[:, p:p+1]
        X_val = val_data[:, 2:1026]
        y_val = val_data[:, p:p+1]

    EPOCH = epoch

    class MyDataset(Dataset):
        def __init__(self, specs, labels):
            self.specs = specs
            self.labels = labels

        def __getitem__(self, index):
            spec, target = self.specs[index], self.labels[index]
            return spec, target

        def __len__(self):
            return len(self.specs)

    def ZspPocessnew(X_train, X_test, X_val, y_train, y_test, y_val):
        global standscale
        global yscaler

        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)
        X_val_Nom = standscale.transform(X_val)

        # yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))
        y_val = yscaler.transform(y_val.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]
        X_val_Nom = X_val_Nom[:, np.newaxis, :]
        ##使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        data_val = MyDataset(X_val_Nom, y_val)
        return data_train, data_test, data_val

    data_train, data_test, data_val = ZspPocessnew(X_train, X_test, X_val, y_train, y_test, y_val)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=TBATCH_SIZE, shuffle=True)

    torch.cuda.empty_cache()
    config = MambaConfig(d_model=1024, n_layers=4)
    model = Mamba(config)

    model1 = model.to(device)

    # 计算参数数量
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Total number of parameters: {total_params}")

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model1.parameters(), lr=LR, weight_decay= 0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, eps=1e-06,
                                                           patience=20)
    train_r2_1 = []
    train_r2_epoch = []
    test_r2_1 = []

    def ModelRgsevaluatePro(y_pred, y_true, yscale):
        yscaler = yscale
        y_true = yscaler.inverse_transform(y_true)
        y_pred = yscaler.inverse_transform(y_pred)

        mse = mean_squared_error(y_true, y_pred)
        R2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        return np.sqrt(mse), R2, mae

    best_r2 = -10
    the_best_rmse = 0
    import time
    epoch_time = []
    for epoch in range(EPOCH):
        start_time = time.time()
        train_losses = []
        model1.train()  # 训练
        train_rmse = []
        train_r2 = []
        train_mae = []
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y

            output = model1(inputs)
            loss = criterion(output, labels)  # MSE
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_losses.append(loss.item())

            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            # plotpred(pred, y_true, yscaler))
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
        avg_train_loss = np.mean(train_losses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        train_r2_epoch.append(epoch)
        train_r2_1.append(avgr2)
        print('Epoch:{}, train: rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

        with torch.no_grad():  # 无梯度
            model1.eval()  # 不训练
            output_val = []
            yture_val = []
            for i, data in enumerate(val_loader):
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
                outputs = model1(inputs)  # 输出等于进入网络后的输入
                pred = outputs.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                output_val.append(pred)
                yture_val.append(y_true)
            # print(output_val)
            # print(yture_val)
            output_val = np.concatenate(output_val, axis=0)
            yture_val = np.concatenate(yture_val, axis=0)
            rmse, R21, mae = ModelRgsevaluatePro(output_val, yture_val, yscaler)
            print('EPOCH:{}, test: rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (rmse), (R21), (mae)))
            scheduler.step(rmse)
            if R21 > best_r2:
                print("****************************best_r2:", best_r2)
                best_r2 = R21
                the_best_rmse = rmse
            # 使用某种方式保存模型，例如保存模型参数或整个模型
            best_model = model1
            # 保存模型到文件，例如使用 joblib 或 pickle
            joblib.dump(best_model, model_path + 'best_model.pkl')
        end_time = time.time()
        run_time = end_time - start_time
        epoch_time.append(run_time)
    #print(sum(epoch_time) / len(epoch_time))

    with torch.no_grad():
        output_val = []
        yture_val = []
        for i, data in enumerate(test_loader):
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
            best_model = joblib.load(model_path + 'best_model.pkl').to(device)
            best_model.eval()
            outputs = best_model(inputs)  # 输出等于进入网络后的输入
            pred = outputs.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            output_val.append(pred)
            yture_val.append(y_true)
        output_val = np.concatenate(output_val, axis=0)
        yture_val = np.concatenate(yture_val, axis=0)
        rmse2, R22, mae = ModelRgsevaluatePro(output_val, yture_val, yscaler)
        y_trues = yscaler.inverse_transform(np.array(yture_val))
        y_preds = yscaler.inverse_transform(np.array(output_val))

        plt.figure()
        plt.scatter(y_trues, y_preds, s=15, alpha=0.7, marker='o')
        ax = plt.gca()
        if min(y_preds) > 0:
            plt.xlim((min(min(y_trues), min(y_preds)) - 0.05, max(max(y_trues), max(y_preds)) + 0.05))
            plt.ylim((min(min(y_trues), min(y_preds)) - 0.05, max(max(y_trues), max(y_preds)) + 0.05))
            plt.plot([0, max(max(y_trues), max(y_preds))[0]], [0, max(max(y_trues), max(y_preds))[0]],
                     linewidth=1.0)
            plt.text((max(max(y_trues), max(y_preds)) - min(min(y_trues), min(y_preds))) * 0.05 + min(min(y_trues),
                                                                                                      min(y_preds))
                     , (max(max(y_trues), max(y_preds)) - min(min(y_trues), min(y_preds))) * 0.8 + min(min(y_trues),
                                                                                                       min(y_preds)),
                     "R2=" + str(round(R22, 2)) + "\n" + "RMSE=" + str(round(rmse2, 2)), fontsize=18,
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
        else:
            plt.xlim((0, max(max(y_trues), max(y_preds)) + 0.05))
            plt.ylim((0, max(max(y_trues), max(y_preds)) + 0.05))
            plt.plot([0, max(max(y_trues), max(y_preds))[0]], [0, max(max(y_trues), max(y_preds))[0]],
                     linewidth=1.0)
            plt.text(max(max(y_trues), max(y_preds)) * 0.05, max(max(y_trues), max(y_preds)) * 0.8,
                     "R2=" + str(round(R22, 2)) + "\n" + "RMSE=" + str(round(rmse2, 2)), fontsize=18,
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
        plt.yticks(size=16)
        plt.xticks(size=16)
        plt.grid()
        plt.plot([0, max(max(y_trues), max(y_preds))[0]], [0, max(max(y_trues), max(y_preds))[0]],
                 linewidth=1.0)
        ax.set_aspect('equal')
        plt.title(name, fontsize=16)
        plt.savefig('../Results/{}.png'.format(name))

    return rmse2, R22, mae
