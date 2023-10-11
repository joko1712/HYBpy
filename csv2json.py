import csv
import pandas as pd
import json
import random

with open('chassbatch1 copy.csv', 'r') as f:
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

    return data


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

    return data


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

    return data


def add_cnoise_to_data(data):
    for batch_key, batch_data in data.items():
        for time_key, time_data in batch_data.items():
            if isinstance(time_data, dict):  # Check if time_data is a dictionary
                cnoise = [float(val) for key, val in time_data.items()
                          if key.startswith("sd")]
                data[batch_key][time_key]['cnoise'] = cnoise
    return data


data = label_batches(data)
data = add_cnoise_to_data(data)

# insert number os batches at the end nbatch
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)
