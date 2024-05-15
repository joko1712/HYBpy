import sys
import os
import json 
import csv
import pandas as pd
import random
import numpy as np
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, manual_label, random_label, add_state_and_time_to_data


with open("sample.json", "r") as f:
    projhyb = json.load(f)


with open("combined_chassbatch.csv", 'r') as f:
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

mode = "2"
data, projhyb = label_batches(data, projhyb, mode)
data = add_state_and_time_to_data(data, projhyb)
count = len(data)
data["nbatch"] = count


# insert number os batches at the end nbatch
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('sample.json', 'w') as f:
    json.dump(projhyb, f, indent=4)
 
with open("file.json", "r") as read_file:
    file = json.load(read_file) 

projhyb, trainData = hybtrain(projhyb, file)
response_data = {
    "message": "Files processed successfully",
    "projhyb": projhyb,  # Assuming this is serializable to JSON
    "trainData": trainData  # Assuming this is serializable to JSON
}
print(projhyb, trainData)