import pandas as pd
import os
import numpy as np
import Preprocessing
from FeatureSelect import FeatureSelect
from Rgs import QuantitativeAnalysis



# File path
def load_data():
    # 指定CSV文件路径
    # 不同国家的数据路径
    path = '../data/raw/LUCAS2015_Soil_Spectra_EU28/'
    # 土壤有机质含量的路径
    csv_file_path = '../data/raw/LUCAS_Topsoil_2015_20200323.csv'
    files = os.listdir(path)
    df_path2 = pd.DataFrame()
    # 拼接不同国家的数据
    for file in files:
        if file.endswith('.csv'):
            df_path2 = pd.concat([df_path2, pd.read_csv(path + file)])

    columns_to_drop = ['source', 'SampleID', 'NUTS_0', 'SampleN']
    df_path2 = df_path2.drop(columns=columns_to_drop)

    result_df = df_path2.groupby('PointID').mean().reset_index()

    # 使用read_csv函数读取CSV文件
    data = pd.read_csv(csv_file_path)
    # ph(CaCL2),ph(H2O),EC,OC,CaCO3,P,N,K
    selected_columns = data.iloc[:, [0, 6, 7, 8, 9, 10, 11, 12, 13]]
    ray_data = result_df

    def targets_index(tg, df):
        tar = np.empty([len(df), 8], dtype='float32', order='C')
        for i in range(len(df)):
            index = tg[tg['Point_ID'] == df['PointID'][i]].index.tolist()[0]
            tar[i] = tg.iloc[index, 1:]
        return tar

    ray_data = ray_data.reset_index(drop=True)
    # draw(ray_data)
    targets = targets_index(selected_columns, ray_data)

    return ray_data.iloc[:, 1:].values, targets


# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, name):
    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定量分析模型, 包括ANN、PLS、SVR、ELM、CNN、SAE等，后续会不断补充完整
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    ProcesedData = Preprocessing.Preprocessing(ProcessMethods, data)
    FeatureData, labels = FeatureSelect.SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label, name)
    Rmse, R2, Mae = QuantitativeAnalysis(FeatureData, labels, name)
    return Rmse, R2, Mae


if __name__ == '__main__':

    if not os.path.exists('data/ray_data.npy') and not os.path.exists('data/targets.npy'):
        ray_data, targets = load_data()
        np.save('../data/ray_data.npy', ray_data)
        np.save('../data/targets.npy', targets)


    ray_data = np.load('../data/ray_data.npy')
    targets = np.load('../data/targets.npy')
    targets_new = targets[:, [0, 1, 3, 4, 6]]
    properties = ['PH(CaCl2)', 'PH(H2O)', 'OC', 'CaCO3', 'N']
    Methods = ['None', 'GA', 'SHAP', 'Lars']
    #for j in range(len(Methods)):
    for j in range(0, 1):
        #for i in range(len(properties)):
        for i in range(4, 5):
            Rmse, R2, Mae = SpectralQuantitativeAnalysis(ray_data[:, :1024], targets_new[:, i], 'SG', Methods[j], properties[i])