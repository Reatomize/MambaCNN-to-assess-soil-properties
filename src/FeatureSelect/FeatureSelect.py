from src.FeatureSelect.Lar import Lar
from src.FeatureSelect.GA import GA
from src.FeatureSelect.SHAP import SHAP


def SpctrumFeatureSelcet(method, X, y, name):
    """
       :param method: 波长筛选/降维的方法，包括：Cars, Lars, Uve, Spa, Pca
       :param X: 光谱数据, shape (n_samples, n_features)
       :param y: 光谱数据对应标签：格式：(n_samples，)
       :return: X_Feature： 波长筛选/降维后的数据, shape (n_samples, n_features)
                y：光谱数据对应的标签, (n_samples，)
    """
    X_Feature = X
    if method == "None":
        X_Feature = X
    elif method == "Lar":
        Featuresecletidx = Lar(X, y, 250)
        X_Feature = X[:, Featuresecletidx]
    elif method == "GA":
        Featuresecletidx = GA(X, y, 250, name)
        X_Feature = X[:, Featuresecletidx]
    elif method == "SHAP":
        Featuresecletidx = SHAP(X, y, 250, name)
        X_Feature = X[:, Featuresecletidx]
    else:
        print("no this method of SpctrumFeatureSelcet!")
    return X_Feature, y