import pandas as pd

def combine_files(file_list, rows_to_add, delimiters):
    combined_df = pd.read_csv(file_list[0], delimiter=delimiters[0]).iloc[:rows_to_add[0]]
    
    combined_df['time'] = combined_df['time'].astype(int)
    
    for file_path, rows, delimiter in zip(file_list[1:], rows_to_add[1:], delimiters[1:]):
        file = pd.read_csv(file_path, delimiter=delimiter)
        
        for col in combined_df.columns:
            if col not in file.columns:
                file[col] = combined_df[col]

        file = file[combined_df.columns]
        
        file['time'] = file['time'].astype(int)
        
        file = file.iloc[:rows]
        
        empty_row = pd.DataFrame({col: [None] for col in combined_df.columns})

        if not file.empty:
            combined_df = pd.concat([combined_df, empty_row, file], ignore_index=True)
    
    return combined_df

file_list = ['TrainningBatchCreation\chassbatch1.csv', 'TrainningBatchCreation\chassbatch2.csv', 'TrainningBatchCreation\chassbatch3.csv', 'TrainningBatchCreation\chassbatch4.csv','TrainningBatchCreation\chassbatch5.csv' ]
rows_to_add = [50, 50, 50, 50, 50]  

delimiters = [',', ';', ',', ',', ','] 

combined_df = combine_files(file_list, rows_to_add, delimiters)

combined_df.to_csv('combined_chass.csv', index=False)