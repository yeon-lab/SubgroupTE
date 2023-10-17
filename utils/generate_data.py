import numpy as np
import pandas as pd

SEED = 1111
np.random.seed(SEED)
    
def Load_split_dataset(config):
    params = config["data_loader"]
    if params['data'] == 'IHDP':
        data = create_IHDP()
    else:
        data = create_synth(params["n_samples"])
    n_samples, n_feat = data[0].shape 
    config["hyper_params"]["input_dim"] = n_feat
        
    n_train, n_test = int(n_samples*params["train_ratio"]), int(n_samples*params["test_ratio"])

    index = np.random.RandomState(seed=SEED).permutation(n_samples)
    train_index = index[:n_train]
    test_index = index[n_train:n_train+n_test]
    valid_index = index[n_train+n_test:]

    train_set = synth_dataset(data, train_index)
    valid_set = synth_dataset(data, valid_index)
    test_set = synth_dataset(data, test_index)
    
    config["data_loader"]["n_samples"] = n_samples
    config["data_loader"]["n_train"] = len(train_index)
    config["data_loader"]["n_valid"] = len(valid_index)
    config["data_loader"]["n_test"] = len(test_index)
    
    return config, train_set, valid_set, test_set


def synth_dataset(data, index):
    (X, T, Y, Y_0, Y_1, Y_cf, TE) = data
    dataset = {
        'X': X[index],
        'T': T[index],
        'Y': Y[index],
        'TE': TE[index]
    }
    return dataset


#######################################################################################################################
# Synthetic data
#######################################################################################################################

def create_synth(n_samples, SEED=1111):
    np.random.seed(seed=SEED)
    X = np.round(np.random.normal(size=(n_samples, 1), loc=66.0, scale=4.1))  # age
    X = np.block([X, np.round(
        np.random.normal(size=(n_samples, 1), loc=6.2, scale=1.0) * 10.0) / 10.0])  # white blood cell count
    X = np.block(
        [X, np.round(np.random.normal(size=(n_samples, 1), loc=0.8, scale=0.1) * 10.0) / 10.0])  # Lymphocyte count
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=183.0, scale=20.4))])  # Platelet count
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=68.0, scale=6.6))])  # Serum creatinine
    X = np.block(
        [X, np.round(np.random.normal(size=(n_samples, 1), loc=31.0, scale=5.1))])  # Aspartete aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=26.0, scale=5.1))])  # Alanine aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=339.0, scale=51))])  # Lactate dehydrogenase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=76.0, scale=21))])  # Creatine kinase
    X = np.block([X, np.floor(np.random.uniform(size=(n_samples, 1)) * 11) + 4])  # Time from study 4~14
    TIME = X[:, 9]

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    T = np.random.binomial(1, 0.5, size=(n_samples,1))

    # sample random coefficients
    coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
    BetaB = np.random.choice(coeffs_, size=9, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])

    MU_0 = np.dot(X[:, 0:9], BetaB)
    MU_1 = np.dot(X[:, 0:9], BetaB)

    logi0 = lambda x: 1 / (1 + np.exp(-(x - 9))) + 5
    logi1 = lambda x: 5 / (1 + np.exp(-(x - 9)))

    MU_0 = MU_0 + logi0(TIME)
    MU_1 = MU_1 + logi1(TIME)

    Y_0 = (np.random.normal(scale=0.1, size=len(X)) + MU_0).reshape(-1,1)
    Y_1 = (np.random.normal(scale=0.1, size=len(X)) + MU_1).reshape(-1,1)

    Y = T * Y_1 + (1 - T) * Y_0
    Y_cf = T * Y_0 + (1 - T) * Y_1
    
    TE = Y_1 - Y_0

    return (X, T, Y, Y_0, Y_1, Y_cf, TE)


def normalize_mean(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (result[feature_name] - result[feature_name].mean()) / result[feature_name].std()
    return result

#######################################################################################################################
# IHDP data
#######################################################################################################################

def create_IHDP(noise=0.1):
    Dataset= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)

    col = ["Treatment", "Response", "Y_CF", "mu0", "mu1", ]

    for i in range(1, 26):
        col.append("X" + str(i))
    Dataset.columns = col
    Dataset.head()

    num_samples = len(Dataset)

    feat_name = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'

    X = np.array(Dataset[feat_name.split()])
    T = np.array(Dataset['Treatment']).reshape(-1,1)

    Y_0 = np.array(np.random.normal(scale=noise, size=num_samples) + Dataset['mu0']).reshape(-1,1)
    Y_1 = np.array(np.random.normal(scale=noise, size=num_samples) + Dataset['mu1']).reshape(-1,1)

    Y = T * Y_1 + (1 - T) * Y_0
    Y_cf = T * Y_0 + (1 - T) * Y_1

    TE = Y_1 - Y_0

    return (X, T, Y, Y_0, Y_1, Y_cf, TE)
    
