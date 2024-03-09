import pandas as pd
from scipy.io import arff

def read_Rice_Cammeo_Osmancik(filepath):
  """
  Rice (Cammeo and Osmancik) - https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
  """
  # 1 - Cammeo
  # 0 - Osmancik
  data, meta = arff.loadarff(filepath)
  df = pd.DataFrame(data)
  df['Class'] = df['Class'].astype(str)
  df['Class'] = df['Class'].replace('Cammeo', 1)
  df['Class'] = df['Class'].replace('Osmancik', 0)
  y = df['Class'].values
  X = df.drop(columns=['Class']).values
  return X, y
  
def read_Online_Shoppers_intention(filepath):
  """
  Online Shoppers intention Dataset - https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
  """
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

  y = df['Revenue'].values
  X = df.drop(columns=['Revenue']).values
  return X, y

def read_Multiple_Disease_Prediction(filepath):
  """
  Multiple Disease Prediction
  https://www.kaggle.com/datasets/ehababoelnaga/multiple-disease-prediction?select=Blood_samples_dataset_balanced_2%28f%29.csv
  """
  # 0 - Healthy
  # 1 - Any disease
  df = pd.read_csv(filepath)
  df['Disease'] = df['Disease'].replace('Healthy', 0)
  df[df['Disease'] != 0] = 1
  y = df['Disease'].values
  X = df.drop(columns=['Disease']).values
  return X, y

def read_Web_Page_Phishing(filepath):
  """
  Web Page Phishing
  https://www.kaggle.com/datasets/danielfernandon/web-page-phishing-dataset
  """
  df = pd.read_csv(filepath)
  y = df['phishing'].values
  X = df.drop(columns=['phishing']).values
  return X, y