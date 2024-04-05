import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the actual data
data_actual = pd.read_csv('chassbatch1.csv')

# Assuming the model predictions are stored in another CSV or added to the DataFrame
# For the example, let's just create some mock predictions within the actual data DataFrame
# In practice, you would replace these lines with loading or generating your model predictions
data_actual['adp_pred'] = data_actual['adp'] * np.random.uniform(0.9, 1.1, len(data_actual))
data_actual['asa_pred'] = data_actual['asa'] * np.random.uniform(0.9, 1.1, len(data_actual))

# Plotting
plt.figure(figsize=(12, 8))

# Actual concentrations vs. Predictions for two species (adp and asa as examples)
plt.errorbar(data_actual['time'], data_actual['adp'], yerr=data_actual['sdadp'], label='ADP Actual', fmt='o', capsize=5)
plt.plot(data_actual['time'], data_actual['adp_pred'], label='ADP Prediction', linestyle='--')

plt.errorbar(data_actual['time'], data_actual['asa'], yerr=data_actual['sdasa'], label='ASA Actual', fmt='o', capsize=5)
plt.plot(data_actual['time'], data_actual['asa_pred'], label='ASA Prediction', linestyle='--')

# Customization
plt.title('Species Concentration: Actual vs. Prediction')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)

# Show plot
plt.show()