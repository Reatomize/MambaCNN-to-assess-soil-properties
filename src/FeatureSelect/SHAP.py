import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import joblib
import shap
import matplotlib.pyplot as plt

LR = 0.001
BATCH_SIZE = 256
TBATCH_SIZE = 256


class MyDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec, target = self.specs[index], self.labels[index]
        return spec, target

    def __len__(self):
        return len(self.specs)


###定义是否需要标准化
def myZspPocessnew(X_train, X_test, y_train, y_test, need=True):  # True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        # yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        ##使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif ((need == False)):
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]

        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        data_train = MyDataset(X_train_new, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test


def SHAP(X, y, nums, name):
    model = joblib.load('../data/processed/' + name + '/' + 'best_model.pkl').to('cpu')
    model.eval()
    rname = "shap_values_" + name
    # Use GPU
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    data_train, data_test = myZspPocessnew(X_train, X_test, y_train, y_test, need=True)

    shap_loader = DataLoader(data_train, batch_size=256, shuffle=True)
    background, _ = next(iter(shap_loader))
    background = background.to(device)
    background = background.float()

    test_loader = DataLoader(data_test, batch_size=256, shuffle=True)
    test_data, _ = next(iter(test_loader))
    test_data = test_data.to(device)
    test_data = test_data.float()
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)

    shap_values = explainer.shap_values(test_data)
    shap_values = np.squeeze(np.array(shap_values))
    #np.save('shap_values/{}.npy'.format(rname), shap_values, allow_pickle=True)
    avg_values = np.array(np.average(shap_values, axis=0))
    avg_values_affect = abs(avg_values)

    avg_values_affect = (avg_values_affect - np.min(avg_values_affect)) / (
            np.max(avg_values_affect) - np.min(avg_values_affect))
    rank = np.argsort(-avg_values_affect)[:nums]
    return rank


def get_topN_reason(old_list, features, top_num=200, min_value=0.0):
    # 输出shap值最高的N个标签
    feature_importance_dict = {}
    for i, f in zip(old_list, features):
        feature_importance_dict[f] = i
    new_dict = dict(sorted(feature_importance_dict.items(), key=lambda e: e[1], reverse=True))
    return_dict = {}
    for k, v in new_dict.items():
        if top_num > 0:
            if v >= min_value:
                return_dict[k] = v
                top_num -= 1
            else:
                break
        else:
            break
    return return_dict
