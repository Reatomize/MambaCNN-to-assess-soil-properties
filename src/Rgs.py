from MambaTrain import MambaTrain

def QuantitativeAnalysis(FeatureData, labels, name):
    Rmse, R2, Mae = MambaTrain(FeatureData, labels, 400, name)

    return Rmse, R2, Mae