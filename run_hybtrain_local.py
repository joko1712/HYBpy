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
#file1_path = "Backend/Files/Workshop/CellGrowthModel3.hmod"
#file2_path = "Backend/Files/Workshop/CellGrowth.csv"
#temp_dir = "Backend/Files/Workshop/Model3"

# CHO
file1_path = "Backend/Files/CHO/CHOsimplenew.hmod"
file2_path = "Backend/Files/CHO/CHOdata.csv"
temp_dir = "Backend/Files/CHO"

user_id = "local_user"
    
run_id = str(uuid.uuid4())
trained_weights = None

#train_batches = [15]
#test_batches = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

train_batches = [9,1,2,3,4,5]
test_batches = [6,7,8]
start_time = time.time()

projhyb = hybdata(file1_path)
projhyb["train_batches"] = train_batches
projhyb["test_batches"] = test_batches
mode = "1"

data = {}
current_time_group = 1
current_time = None

with open(file2_path, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)

    data = {}
    current_time_group = 1
    current_time = None

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

if mode == "1":
    train_batches = list(map(int, train_batches))
    test_batches = list(map(int, test_batches))
    for batch in train_batches:
        if batch in data:
            data[batch]["istrain"] = 1
    for batch in test_batches:
        if batch in data:
            data[batch]["istrain"] = 3

    all_batches = set(data.keys())
    for batch in all_batches:
        if batch not in train_batches and batch not in test_batches:
            data[batch]["istrain"] = 0

    projhyb["train_batches"] = train_batches
    projhyb["test_batches"] = test_batches

if mode == "2":
    data, projhyb = label_batches(data, projhyb, mode)

print(projhyb["train_batches"])
print(projhyb["test_batches"])

data = add_state_and_time_to_data(data, projhyb)
count = len(data)
data["nbatch"] = count


data_json_path = os.path.join(temp_dir, "data.json")
with open(data_json_path, "w") as write_file:
    json.dump(data, write_file)

projhyb_json_path = os.path.join(temp_dir, "projhyb.json")
with open(projhyb_json_path, "w") as write_file:
    json.dump(projhyb, write_file)



projhyb, bestWeights, testing, newHmodFile = hybtrain(
    projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_id
)

endTime = time.time()

print(f"⏱️ Training Time: {endTime - start_time:.2f} seconds")
print("\n✅ Training Complete")
print(f"Best Weights (first 5): {bestWeights[:5]}")
print(f"Metrics: {testing}")
print(f"Saved HMOD File: {newHmodFile}")
