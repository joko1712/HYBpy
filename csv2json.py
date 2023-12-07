import csv
import pandas as pd
import json
import random
import numpy as np

with open("sample.json", "r") as f:
    projhyb = json.load(f)

with open('chassbatch1.csv', 'r') as f:
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


def label_batches(data):
    print(f"Choose batches selection mode:")
    print(f"1. Manual")
    print(f"2. Random")

    mode = input("Enter mode: ")
    if mode == "1":
        return manual_label(data)
    elif mode == "2":
        return random_label(data)
    else:
        print("Invalid mode. Exiting...")
        exit()

    return data, projhyb


def manual_label(data):
    all_batches = list(data.keys())
    print(f"Available batches: {all_batches}")
    train_batches = input(
        "Enter batches for training (comma separated): ").split(",")
    test_batches = input(
        "Enter batches for testing (comma separated): ").split(",")

    if not train_batches[0] and not test_batches[0]:
        print("Invalid input. Exiting...")
        exit()

    for batch in train_batches:
        if batch in data:
            data[batch]["istrain"] = 1
    for batch in test_batches:
        if batch in data:
            data[batch]["istrain"] = 3

    projhyb["train_batches"] = train_batches
    projhyb["test_batches"] = test_batches

    return data, projhyb


def random_label(data):
    all_batches = list(data.keys())
    random.shuffle(all_batches)
    split_idx = int(len(all_batches) * 2 / 3)
    train_batches = all_batches[:split_idx]
    test_batches = all_batches[split_idx:]

    for batch in train_batches:
        if batch in data:
            data[batch]["istrain"] = 1
    for batch in test_batches:
        if batch in data:
            data[batch]["istrain"] = 3

    projhyb["train_batches"] = train_batches
    projhyb["test_batches"] = test_batches

    return data, projhyb


def add_state_and_time_to_data(data):
    for batch_key, batch_data in data.items():
        time_list = []
        y_matrix = []
        sc_list = []
        sy_matrix = []
        for time_key, time_data in batch_data.items():
            if isinstance(time_data, dict):
                state = [float(val) for key, val in time_data.items() if not key.startswith("sd")]

                data[batch_key][time_key]['state'] = state
                y_matrix.extend(state) 
                
                sd_values = [float(val) for key, val in time_data.items() if key.startswith("sd")]
                v_values = [float(time_data[key]) for key in time_data if key.startswith("V")]

                data[batch_key][time_key]['sc'] = sd_values + [np.std(v_values)]
                sy_matrix.extend(data[batch_key][time_key]['sc'])
                                
                time_list.append(int(time_key)) 

        data[batch_key]['time'] = time_list
        data[batch_key]["np"] = len(time_list)
        data[batch_key]["y"] = y_matrix 
        data[batch_key]["sy"] = sy_matrix

    return data



data, projhyb = label_batches(data)
data = add_state_and_time_to_data(data)
count = len(data)
data["nbatch"] = count


# insert number os batches at the end nbatch
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('sample.json', 'w') as f:
    json.dump(projhyb, f, indent=4)


#TODO: 
''' for k=1:projhyb.batch(i).np-1
        projhyb.batch(i).rnoise(k,:)=...
        (projhyb.batch(i).cnoise(k+1,:)-projhyb.batch(i).cnoise(k,:))...
            /(projhyb.batch(i).t(k+1)-projhyb.batch(i).t(k));
    end
'''