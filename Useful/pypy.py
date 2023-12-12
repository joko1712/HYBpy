import pandas as pd
import numpy as np

df = pd.read_csv('chassbatch1 copy.csv')

time_data = ['Time']
time_data.extend(df.iloc[:, 0].tolist())

odd_columns_data = [df.columns[1::2].tolist()]
odd_columns_data.extend(df.iloc[:, 1::2].values.tolist())

even_columns_data = [df.columns[2::2].tolist()]
even_columns_data.extend(df.iloc[:, 2::2].values.tolist())


print("Time data:", time_data)
print("Odd columns data:", odd_columns_data)
print("Even columns data:", even_columns_data)


time_matrix = np.array(time_data)
odd_columns_matrix = np.array(odd_columns_data)
even_columns_matrix = np.array(even_columns_data)


print("Time matrix:", time_matrix)
print("Odd columns matrix:", odd_columns_matrix)
print("Even columns matrix:", even_columns_matrix)
