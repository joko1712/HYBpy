import json
import os
import tempfile
import uuid
import csv
import time

from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, add_state_and_time_to_data

# Model 1
#file1_path = "Backend/Files/Workshop/CellGrowthModel1.hmod"
#file2_path = "Backend/Files/Workshop/CellGrowth.csv"
#temp_dir = "Backend/Files/Workshop/Model1"


# Model 2
#file1_path = "Backend/Files/Workshop/CellGrowthModel2.hmod"
#file2_path = "Backend/Files/Workshop/CellGrowth.csv"
#temp_dir = "Backend/Files/Workshop/Model2"

# Model 3
file1_path = "Backend/Files/Workshop/CellGrowthModel3.hmod"
file2_path = "Backend/Files/Workshop/CellGrowth.csv"
temp_dir = "Backend/Files/Workshop/Model3"

user_id = "local_user"
train_batches = [15]    
test_batches = [1, 2, 3, 5, 10, 8, 15, 12, 9, 7, 4, 6, 13, 14, 11]        
run_id = str(uuid.uuid4())
trained_weights = None

start_time = time.time()

projhyb = hybdata(file1_path)
projhyb["train_batches"] = train_batches
projhyb["test_batches"] = test_batches

data = {}
current_time_group = 1
current_time = None

with open(file2_path, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)

    for row in reader:
        if not row or not row[0].strip():
            current_time_group += 1
            continue
        current_time = row[0]

        if current_time_group not in data:
            data[current_time_group] = {}
        if current_time not in data[current_time_group]:
            data[current_time_group][current_time] = {}
        for h, value in zip(headers[1:], row[1:]):
            data[current_time_group][current_time][h] = value

data, projhyb = label_batches(data, projhyb, mode="2")
data = add_state_and_time_to_data(data, projhyb)
data["nbatch"] = len(data)

#temp_dir = tempfile.mkdtemp()
data_json_path = os.path.join(temp_dir, "data.json")


with open(data_json_path, "w") as f:
    json.dump(data, f)

projhyb, bestWeights, testing, newHmodFile = hybtrain(
    projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_id
)

endTime = time.time()

print(f"⏱️ Training Time: {endTime - start_time:.2f} seconds")
print("\n✅ Training Complete")
print(f"Best Weights (first 5): {bestWeights[:5]}")
print(f"Metrics: {testing}")
print(f"Saved HMOD File: {newHmodFile}")
