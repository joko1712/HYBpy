# THE SERVER WILL RUN ON http://localhost:5000
from flask import Flask, request
from flask_cors import CORS
import sys
import os
import json
import csv
import pandas as pd
import json
import random
import numpy as np
from dotenv import load_dotenv

 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, manual_label, random_label, add_state_and_time_to_data


import firebase_admin
from firebase_admin import credentials, firestore, storage

load_dotenv()

cred = credentials.Certificate("../hybpy-test-firebase-adminsdk-20qxj-245fd03d89.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("STORAGE_BUCKET_NAME")
})

db = firestore.client()

app = Flask(__name__)
CORS(app)

def upload_file_to_storage(file, user_id, filename):
    file.seek(0)
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))
    blob = bucket.blob(f"users/{user_id}/{filename}")
    blob.upload_from_file(file, content_type=file.content_type)

    blob.make_public()

    return blob.public_url

# This is the request http://localhost:5000/upload
@app.route('/upload', methods=['POST'])
def upload_file():

    while os.path.exists("file.json"):
        print("file.json exists")
        os.remove("file.json")

    while os.path.exists("sample.json"):
        print("sample.json exists")
        os.remove("sample.json")

    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    mode = request.form.get('mode')
    user_id = request.form.get('userId')
    description = request.form.get('description')
    train_batches = request.form.get('train_batches').split(",")
    test_batches = request.form.get('test_batches').split(",")

    if not file1 or not file2:
        return {"error": "Both files are required"}, 400
    if file1 and file2:
        file1.save(file1.filename)
        file2.save(file2.filename)

        file1_url = upload_file_to_storage(file1, user_id, file1.filename)
        file2_url = upload_file_to_storage(file2, user_id, file2.filename)

        hybdata(file1.filename)
        print("file1", file1.filename)
        print("file2", file2.filename)

        with open("sample.json", "r") as f:
            projhyb = json.load(f)

        with open(file2.filename, 'r') as f:
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
            for batch in train_batches:
                if batch in data:
                    data[batch]["istrain"] = 1
            for batch in test_batches:
                if batch in data:
                    data[batch]["istrain"] = 3

            projhyb["train_batches"] = train_batches
            projhyb["test_batches"] = test_batches

        if mode == "2":
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

        #projhyb, trainData = hybtrain(projhyb, file)
        trainData = "trainData"
        response_data = {
            "message": "Files processed successfully",
            "projhyb": projhyb,
            "trainData": trainData 
        }
        add_run(file1_url, file2_url, file1.filename, file2.filename, response_data, user_id, description, mode)
        return json.dumps(response_data), 200

        if trainData is None:
            return {"message": "File not found"}, 400
        print(projhyb, trainData)

    else:
        return {"message": "File not found"}, 400


@app.route('/add-run', methods=['POST'])
def add_run(file1_url, file2_url, file1, file2, response_data, user_id, description, mode):
    run_data = {
        "userId": user_id,
        "file1": file1_url,
        "file2": file2_url,
        "file1_name": file1,
        "file2_name": file2,
        "response_data": response_data,
        "description": description,
        "mode": mode,
        "createdAt": firestore.SERVER_TIMESTAMP
    }
    user_ref = db.collection('users').document(user_id)
    ren_ref = user_ref.collection('runs').document()
    ren_ref.set(run_data)
    return json.dumps(response_data), 200


@app.route("/get-available-batches", methods=['POST'])
def get_available_batches():

    file2 = request.files.get('file2')
    

    if file2: 

        file2.save(file2.filename)

        with open(file2.filename, 'r') as f:
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
        
        all_batches = list(data.keys())
        print(all_batches)

        return json.dumps(all_batches), 200
    
    else:
        return {"message": "File not found"}, 400


if __name__ == '__main__':
    app.run(debug=True)

