import pandas as pd

# Load the CSV files
chassbatch1 = pd.read_csv('chassbatch1.csv')
chassbatch2 = pd.read_csv('chassbatch2.csv')

chassbatch1_split = chassbatch1.iloc[:, 0].str.split(';', expand=True)
chassbatch2_split = chassbatch2.iloc[:, 0].str.split(';', expand=True)

# Rename the columns to match those of chassbatch1
chassbatch1_split.columns = [
    'time', 'adp', 'sdadp', 'asa', 'sdasa', 'asp', 'sdasp', 'aspp', 'sdaspp',
    'atp', 'sdatp', 'hs', 'sdhs', 'hsp', 'sdhsp', 'nadp', 'sdnadp', 'nadph',
    'sdnadph', 'phos', 'sdphos', 'thr', 'sdthr', 'V', 'sdV'
]
chassbatch2_split.columns = [
    'time', 'adp', 'asa', 'asp', 'aspp', 'atp', 'hs', 'hsp', 'nadp', 'nadph',
    'phos', 'thr'
]

# Convert all columns to string type for compatibility
chassbatch1_split = chassbatch1_split.astype(str)
chassbatch2_split = chassbatch2_split.astype(str)

# Create an empty DataFrame with the same columns as chassbatch1_split
empty_row = pd.DataFrame([[''] * len(chassbatch1_split.columns)], columns=chassbatch1_split.columns)

# Concatenate chassbatch1_split, the empty row, and chassbatch2_split
combined = pd.concat([chassbatch1_split, empty_row, chassbatch2_split], ignore_index=True)

# Save the combined result to a new CSV file
output_file = 'combined_chassbatch.csv'
combined.to_csv(output_file, index=False, header=False)

print(f"Combined file saved to {output_file}")