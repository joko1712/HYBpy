import csv
import pandas as pd
import json
import random
import numpy as np

'''
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
'''

def label_batches(data, projhyb, mode):
    print(f"Choose batches selection mode:")
    print(f"1. Manual")
    print(f"2. Random")

    if mode == "1":
        return manual_label(data, projhyb)
    elif mode == "2":
        return random_label(data, projhyb)
    else:
        print("Invalid mode. Exiting...")
        exit()

    return data, projhyb


def manual_label(data, projhyb):
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

def label_batches(data, projhyb, mode, use_validation=False, val_ratio=0.3, kfolds=1, manual_test_batches=None):
    all_batches = list(data.keys())
    kfolds = int(kfolds) 

    for b in all_batches:
        data[b]["istrain"] = [0] * kfolds

    shuffled = all_batches[:]
    random.shuffle(shuffled)

    if manual_test_batches:
        global_test_batches = manual_test_batches
        remaining_batches = [b for b in shuffled if b not in global_test_batches]
    else:
        if use_validation:
            test_ratio = 1.0 - (1.0 - val_ratio)  
            train_ratio = 1.0 - val_ratio - test_ratio
            n = len(shuffled)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            global_test_batches = shuffled[val_end:]
            remaining_batches = shuffled[:val_end]
        else:
            split_idx = int(len(shuffled) * (1 - val_ratio))  
            global_test_batches = shuffled[split_idx:]
            remaining_batches = shuffled[:split_idx]

    kfolds_train = []
    kfolds_val = []
    used_splits = set()

    for k in range(kfolds):
        while True:
            rem_shuffled = remaining_batches[:]
            random.shuffle(rem_shuffled)

            if use_validation:
                n = len(rem_shuffled)
                train_end = int(n * (1 - val_ratio))
                val_end = n 
                train_batches = rem_shuffled[:train_end]
                val_batches = rem_shuffled[train_end:val_end]
            else:
                train_batches = rem_shuffled
                val_batches = []

            split_signature = (tuple(sorted(train_batches)), tuple(sorted(val_batches)))
            if split_signature not in used_splits:
                used_splits.add(split_signature)
                break

        kfolds_train.append(train_batches)
        kfolds_val.append(val_batches)

        for b in train_batches:
            data[b]["istrain"][k] = 1
        for b in val_batches:
            data[b]["istrain"][k] = 2
        for b in global_test_batches:
            data[b]["istrain"][k] = 3

    for b in all_batches:
        for k in range(kfolds):
            if data[b]["istrain"][k] == 0:
                if b in global_test_batches:
                    data[b]["istrain"][k] = 3
                else:
                    data[b]["istrain"][k] = 1

    projhyb["kfolds_splits"] = [
        {"train": tr, "val": vl} for tr, vl in zip(kfolds_train, kfolds_val)
    ]
    projhyb["global_test_batches"] = global_test_batches
    projhyb["nkfolds"] = kfolds

    return data, projhyb


def add_state_and_time_to_data(data, projhyb):
    for batch_key, batch_data in data.items():
        time_list = []
        y_matrix = []
        sc_list = []
        sy_matrix = []
        key_value_list = []
        for time_key, time_data in batch_data.items():
            if isinstance(time_data, dict):
                state = [float(val) for key, val in time_data.items() if key != "time" and not key.startswith("sd") and not isinstance(val, list)]
                y_matrix.append(state)

                sd_values = [float(val) for key, val in time_data.items() if key.startswith("sd")]
                sy_matrix.append(sd_values)
                
                time_list.append(int(time_key))
                
                key_value_dict = {}
                for key, val in time_data.items():
                    if key != "time" and not key.startswith("sd"):
                        try:
                            key_value_dict[key] = float(val)
                        except (ValueError, TypeError):
                            key_value_dict[key] = val 
                key_value_list.append(key_value_dict)

        data[batch_key]['time'] = time_list
        data[batch_key]["np"] = len(time_list)
        data[batch_key]["y"] = y_matrix
        data[batch_key]["sy"] = sy_matrix
        data[batch_key]["key_value_array"] = key_value_list

    return data


'''
data, projhyb = label_batches(data)
data = add_state_and_time_to_data(data)
count = len(data)
data["nbatch"] = count


# insert number os batches at the end nbatch
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('sample.json', 'w') as f:
    json.dump(projhyb, f, indent=4)
'''

#TODO: 
''' for k=1:projhyb.batch(i).np-1
        projhyb.batch(i).rnoise(k,:)=...
        (projhyb.batch(i).cnoise(k+1,:)-projhyb.batch(i).cnoise(k,:))...
            /(projhyb.batch(i).t(k+1)-projhyb.batch(i).t(k));
    end
'''


