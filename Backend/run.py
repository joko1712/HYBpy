import sys
import os
import json 
import csv
import pandas as pd
import random
import numpy as np
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, manual_label, random_label, add_state_and_time_to_data

projhyb = hybdata("Backend/Chass1.hmod")



with open("Backend/combined_chass.csv", 'r') as f:
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

with open("file.json", "w") as write_file:
    json.dump(data, write_file)
    
'''
with h5py.File('file.h5', 'w') as f:
    for key, value in data.items():
        f.create_dataset(key, data=value)

with h5py.File('sample.h5', 'w') as f:
    for key, value in data.items():
        f.create_dataset(key, data=value)

with open("file.json", "r") as read_file:
    file = json.load(read_file) 
'''
user_id = 'Joko'

projhyb, trainData = hybtrain(projhyb, data, user_id)
response_data = {
    "message": "Files processed successfully",
    "projhyb": projhyb,
    "trainData": trainData 
}
