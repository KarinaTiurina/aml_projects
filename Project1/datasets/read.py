import pandas as pd
from scipy.io import arff


def read_Rice_Cammeo_Osmancik():
    """
    Rice (Cammeo and Osmancik) - https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    """
    # 1 - Cammeo
    # 0 - Osmancik
    data, meta = arff.loadarff('data/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff')
    df = pd.DataFrame(data)
    df['Class'] = df['Class'].astype(str)
    df['Class'] = df['Class'].replace('Cammeo', 1)
    df['Class'] = df['Class'].replace('Osmancik', 0)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    y = df['Class'].values
    X = df.drop(columns=['Class']).values
    return X, y


def read_Online_Shoppers_intention():
    """
    Online Shoppers intention Dataset - https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
    """
    filepath = 'data/online+shoppers+purchasing+intention+dataset/online_shoppers_intention.csv'
    df = pd.read_csv(filepath, sep=',', encoding='utf-8')
    df['Month'] = df['Month'].replace('Feb', 2)
    df['Month'] = df['Month'].replace('Mar', 3)
    df['Month'] = df['Month'].replace('May', 5)
    df['Month'] = df['Month'].replace('June', 6)
    df['Month'] = df['Month'].replace('Jul', 7)
    df['Month'] = df['Month'].replace('Aug', 8)
    df['Month'] = df['Month'].replace('Sep', 9)
    df['Month'] = df['Month'].replace('Oct', 10)
    df['Month'] = df['Month'].replace('Nov', 11)
    df['Month'] = df['Month'].replace('Dec', 12)
    df['VisitorType'] = df['VisitorType'].replace('Returning_Visitor', 1)
    df['VisitorType'] = df['VisitorType'].replace('New_Visitor', 2)
    df['VisitorType'] = df['VisitorType'].replace('Other', 0)
    df['Weekend'] = df['Weekend'].replace(False, 0)
    df['Weekend'] = df['Weekend'].replace(True, 1)
    df['Revenue'] = df['Revenue'].replace(False, 0)
    df['Revenue'] = df['Revenue'].replace(True, 1)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df['Revenue'].values
    X = df.drop(columns=['Revenue']).values
    return X, y


def read_Multiple_Disease_Prediction():
    """
    Multiple Disease Prediction
    https://www.kaggle.com/datasets/ehababoelnaga/multiple-disease-prediction?select=Blood_samples_dataset_balanced_2%28f%29.csv
    """
    filepath = 'data/Multiple Disease Prediction/Blood_samples_dataset_balanced_2(f).csv'
    # 0 - Healthy
    # 1 - Any disease
    df = pd.read_csv(filepath)
    df['Disease'] = df['Disease'].replace('Healthy', 0)
    df[df['Disease'] != 0] = 1

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.95:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # do not remove the target column
    collinear_features.remove('Disease')
    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df['Disease'].values
    X = df.drop(columns=['Disease']).values
    return X, y


def read_Web_Page_Phishing():
    """
    Web Page Phishing
    https://www.kaggle.com/datasets/danielfernandon/web-page-phishing-dataset
    """
    filepath = 'data/Web Page Phishing/web-page-phishing.csv'
    df = pd.read_csv(filepath)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df['phishing'].values
    X = df.drop(columns=['phishing']).values
    return X, y


def read_Dataset_for_Link_Phishing():
    """
    Dataset for Link Phishing
    https://www.kaggle.com/datasets/winson13/dataset-for-link-phishing-detection
    """
    filepath = 'data/Dataset for Link Phishing/dataset_link_phishing.csv'
    df = pd.read_csv(filepath, low_memory=False)
    df = df.drop(columns=['Unnamed: 0', 'url'])
    df['domain_with_copyright'].unique()
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('one', 1)
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('One', 1)
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('zero', 0)
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('Zero', 0)
    df['status'] = df['status'].replace('phishing', 1)
    df['status'] = df['status'].replace('legitimate', 0)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df['status'].values
    X = df.drop(columns=['status']).values
    return X, y


def read_Statlog_Shuttle():
    """
    Statlog (Shuttle) Data Set
    https://archive.ics.uci.edu/dataset/148/statlog+shuttle
    """
    filepaths = ['data/statlog+shuttle/shuttle.trn', 'data/statlog+shuttle/shuttle.tst']
    df1 = pd.read_csv(filepaths[0], sep=' ', header=None)
    df2 = pd.read_csv(filepaths[1], sep=' ', header=None)
    df = pd.concat([df1, df2])

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()
    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df[9].values
    X = df.drop(columns=[9]).values

    # Remap the labels: 1 - first class, 0 - other classes
    y[y != 1] = 0

    return X, y


def read_Banknote_Authentication():
    """
    Banknote Authentication Data Set
    https://archive.ics.uci.edu/dataset/267/banknote+authentication
    """
    filepath = 'data/banknote+authentication/data_banknote_authentication.txt'
    df = pd.read_csv(filepath, sep=',', header=None)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df[4].values
    X = df.drop(columns=[4]).values

    return X, y


def read_Airline_Passenger_Satisfaction():
    """
    Airline Passenger Satisfaction
    https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
    """
    filepaths = ['data/Airline Passenger Satisfaction/train.csv', 'data/Airline Passenger Satisfaction/test.csv']
    df1 = pd.read_csv(filepaths[0])
    df2 = pd.read_csv(filepaths[1])

    df = pd.concat([df1, df2])

    # drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'id'])

    # Encode the categorical features
    df['Gender'] = df['Gender'].replace('Male', 1)
    df['Gender'] = df['Gender'].replace('Female', 0)

    df['Customer Type'] = df['Customer Type'].replace('Loyal Customer', 1)
    df['Customer Type'] = df['Customer Type'].replace('disloyal Customer', 0)

    df['Type of Travel'] = df['Type of Travel'].replace('Personal Travel', 1)
    df['Type of Travel'] = df['Type of Travel'].replace('Business travel', 0)

    df['Class'] = df['Class'].replace('Eco Plus', 2)
    df['Class'] = df['Class'].replace('Business', 1)
    df['Class'] = df['Class'].replace('Eco', 0)

    df['satisfaction'] = df['satisfaction'].replace('neutral or dissatisfied', 0)
    df['satisfaction'] = df['satisfaction'].replace('satisfied', 1)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()

    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df['satisfaction'].values
    X = df.drop(columns=['satisfaction']).values
    return X, y


def read_Optdigits():
    """
    Optical Recognition of Handwritten Digits Data Set
    https://www.openml.org/search?type=data&sort=qualities.NumberOfNumericFeatures&status=active&order=desc&qualities.NumberOfFeatures=between_10_100&qualities.NumberOfClasses=%3D_2&id=980
    """
    filepath = 'data/optdigits/optdigits.arff'
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Remap the labels: 1 - Positive, 0 - Negative
    df['binaryClass'] = df['binaryClass'].replace(b'P', 1)
    df['binaryClass'] = df['binaryClass'].replace(b'N', 0)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    # find the collinear features
    corr = df.corr()
    collinear_features = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) > 0.8:
                collinear_features.add(df.columns[i])
                collinear_features.add(df.columns[j])

    # remove the collinear features
    df = df.drop(columns=list(collinear_features))

    y = df['binaryClass'].values
    X = df.drop(columns=['binaryClass']).values
    return X, y
