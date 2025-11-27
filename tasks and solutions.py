#DATA
import numpy as np
X=np.array([
    [14.23, 1.71, 2.43, 15.6, 127],
    [13.20, 1.78, 2.14, 11.2, 100],
    [13.16, 2.36, 2.67, 18.6, 101],
    [14.37, 1.95, 2.50, 16.8, 113],
    [13.24, 2.59, 2.87, 21.0, 118],
    [14.20, 1.76, 2.45, 15.2, 112],
    [14.39, 1.87, 2.45, 14.6, 96],
    [14.06, 2.15, 2.61, 17.6, 121],
    [14.83, 1.64, 2.17, 14.0, 97],
    [13.86, 1.35, 2.27, 16.0, 98]
])
# Task1
num_rows,num_columns=X.shape
print(f"X's shape: {num_rows,num_columns}")
#Task2
mean1=np.mean(X,axis=0)
median1=np.median(X,axis=0)
std1=np.std(X,axis=0)
print("Mean:",mean1)
print("Median:",median1)
print("Std:",std1)

#Task3
def normalize_columns(arr):
    col_min=np.min(arr,axis=0)
    col_max=np.max(arr,axis=0)
    normalized=(arr-col_min)/(col_max-col_min)
    return normalized

normal_X=normalize_columns(X)
print(normal_X)
# Task 4(Alcohol>13 and Malic acid < 2)
selected_rows = X[(X[:, 0] > 13)&(X[:, 1] < 2)]
print("(Alcohol > 13 & Malic acid < 2):\n",selected_rows)
print("Number of selected rows:",selected_rows.shape[0])
#Task5
corr_matrix=np.corrcoef(X,rowvar=False)
print(corr_matrix)
#Task6
def standardize(arr):
    mean=np.mean(arr,axis=0)
    std=np.std(arr,axis=0)
    standardized=(arr-mean)/std
    return standardized
standardized_X=standardize(X)
print("\nStandardized array:\n",standardized_X)
print("Column means after standardization~0:",np.mean(standardized_X,axis=0))
print("Column stds after standardization:~1",np.std(standardized_X,axis=0))