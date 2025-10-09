import json
import os
import tempfile
import uuid
import csv
import time

from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, add_state_and_time_to_data

start_time = time.time()
user_id = "local_user"
    
run_id = str(uuid.uuid4())
trained_weights = None

# Model 1
file1_path = "Backend/Files/Workshop/CellGrowthModel1.hmod"
file2_path = "Backend/Files/Workshop/CellGrowth.csv"
temp_dir = "Backend/Files/Workshop/CellGrowthModel1"

# Model 2
#file1_path = "Backend/Files/Workshop/CellGrowthModel2.hmod"
#file2_path = "Backend/Files/Workshop/CellGrowth.csv"
#temp_dir = "Backend/Files/Workshop/CellGrowthModel2"

# Model 3
#file1_path = "Backend/Files/Workshop/CellGrowthModel3.hmod"
#file2_path = "Backend/Files/Workshop/CellGrowth.csv"
#temp_dir = "Backend/Files/Workshop/CellGrowthModel3"

# CHO
#file1_path = "Backend/Files/CHO/CHOsimplenew.hmod"
#file2_path = "Backend/Files/CHO/CHOdata.csv"
#temp_dir = "Backend/Files/CHO"

#Chassagnole
#file1_path = "Backend/Files/Chassagnole/Chass1.hmod"
#file2_path = "Backend/Files/Chassagnole/combined_chass_50.csv"
#temp_dir = "Backend/Files/Chassagnole"

#ParkAndRamirez
#file1_path = "Backend/Files/ParkAndRamirez/parkramstandard.hmod"
#file2_path = "Backend/Files/ParkAndRamirez/PARK_COMBINED_PM_PT_S_X.csv"
#temp_dir = "Backend/Files/ParkAndRamirez"

# RUN Manual Hold-Out Cross Validation:

train_batches = [1,2,3,4,5,6,7,8,9,10,11,13,14]
test_batches = [15]
val_batches = [12]

mode = "1"
Kfolds = 1
nensemble = 1
split_ratio = 0.5
Crossval = "1"

# RUN Automatic Hold-Out Cross Validation
'''
train_batches = []
test_batches = [13,14,15]
val_batches = []


mode = "2"
Kfolds = 1
nensemble = 1
split_ratio = 0.5
Crossval = "1"
'''

# RUN K-Fold Cross Validation
'''
test_batches = [13,14,15]
train_batches = []
val_batches = []

mode = "3"
Kfolds = 5
nensemble = 3
split_ratio = 0.5
Crossval = "1"
'''

projhyb = hybdata(file1_path)


use_validation = True if Crossval == "1" else False
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
    val_batches = list(map(int, val_batches))
    for batch in train_batches:
        if int(batch) in data:
            data[int(batch)]["istrain"] = [1]  

    for batch in val_batches:
        if int(batch) in data:
            data[int(batch)]["istrain"] = [2]

    for batch in test_batches:
        if int(batch) in data:
            data[int(batch)]["istrain"] = [3]

    all_batches = set(data.keys())
    for batch in all_batches:
        if int(batch) not in train_batches and int(batch) not in test_batches and int(batch) not in val_batches:
            data[int(batch)]["istrain"] = [0]

if mode == "2" or mode == "3":
    test_batches = list(map(int, test_batches))
    kfolds = int(Kfolds) if Kfolds else 1
    use_validation = Crossval == "1"

    data, projhyb = label_batches(
        data,
        projhyb,
        mode,
        use_validation=use_validation,
        val_ratio=1 - float(split_ratio),
        kfolds=kfolds,
        manual_test_batches=test_batches
    )
    


data = add_state_and_time_to_data(data, projhyb)
count = len(data)
data["nbatch"] = count


data_json_path = os.path.join(temp_dir, "data.json")
with open(data_json_path, "w") as write_file:
    json.dump(data, write_file)

projhyb['kfolds'] = Kfolds
projhyb['nensemble'] = nensemble
projhyb["train_batches"] = train_batches
projhyb["test_batches"] = test_batches
projhyb["val_batches"] = val_batches
projhyb["crossval"] = Crossval
projhyb["split_ratio"] = split_ratio
print("train batches:", train_batches)
print("test batches:", test_batches)
print("val batches:", val_batches)

projhyb_json_path = os.path.join(temp_dir, "projhyb.json")
with open(projhyb_json_path, "w") as write_file:
    json.dump(projhyb, write_file)


projhyb, bestWeights, testing, newHmodFile = hybtrain(
    projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_id
)

endTime = time.time()

print(f"⏱️ Training Time: {endTime - start_time:.2f} seconds")
print("\n✅ Training Complete")
print(f"Metrics: {testing}")
