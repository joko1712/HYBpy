import pandas as pd

def combine_files(file_list, rows_to_add, delimiters):
    # Load the first file with the specified number of rows and delimiter
    combined_df = pd.read_csv(file_list[0], delimiter=delimiters[0]).iloc[:rows_to_add[0]]
    
    # Ensure the time column is an integer
    combined_df['time'] = combined_df['time'].astype(int)
    
    for file_path, rows, delimiter in zip(file_list[1:], rows_to_add[1:], delimiters[1:]):
        # Load the current file with the specified delimiter
        file = pd.read_csv(file_path, delimiter=delimiter)
        
        # Add missing columns to the current file by copying the entire column from the first file
        for col in combined_df.columns:
            if col not in file.columns:
                file[col] = combined_df[col]

        # Ensure the columns of the current file match the order of the combined DataFrame
        file = file[combined_df.columns]
        
        # Ensure the time column is an integer
        file['time'] = file['time'].astype(int)
        
        # Select the specified number of rows
        file = file.iloc[:rows]
        
        # Create an empty row with the same columns
        empty_row = pd.DataFrame({col: [None] for col in combined_df.columns})

        # Check if the current file is empty, if not, add the empty row and file data
        if not file.empty:
            combined_df = pd.concat([combined_df, empty_row, file], ignore_index=True)
    
    return combined_df

# List of files to combine
file_list = ['TrainningBatchCreation\chassbatch1.csv', 'TrainningBatchCreation\chassbatch2.csv', 'TrainningBatchCreation\chassbatch3.csv', 'TrainningBatchCreation\chassbatch4.csv','TrainningBatchCreation\chassbatch5.csv' ]
# Number of rows to add from each file (including the first one)
rows_to_add = [46, 46, 46, 46, 46]  # Example: 10 rows from file1, 10 rows from file2, and 15 rows from file3

delimiters = [',', ';', ',', ',', ',']  # Example: ',' for file1, ';' for file2, and ',' for file3

# Combine the files
combined_df = combine_files(file_list, rows_to_add, delimiters)

# Save the combined data to a new CSV file
combined_df.to_csv('combined_chass.csv', index=False)