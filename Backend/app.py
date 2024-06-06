from flask import Flask, request
from flask_cors import CORS
import sys
import os
import json
import csv
import pandas as pd
import random
import numpy as np
from dotenv import load_dotenv
import time
import glob
import matplotlib
import logging
import uuid
import shutil
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, manual_label, random_label, add_state_and_time_to_data
import re
import firebase_admin
from firebase_admin import credentials, firestore, storage

load_dotenv()

cred = credentials.Certificate("../hybpy-test-firebase-adminsdk-20qxj-245fd03d89.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "hybpy-test.appspot.com"
})

db = firestore.client()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

def upload_file_to_storage(file, user_id, filename, folder_id):
    file.seek(0)
    bucket = storage.bucket("hybpy-test.appspot.com")
    blob = bucket.blob(f"{user_id}/{folder_id}/{filename}")
    blob.upload_from_file(file, content_type=file.content_type)
    blob.make_public()
    return blob.public_url

def upload_plots_to_gcs(user_id, folder_id):
    plot_urls = []
    user_dir = os.path.join('plots', user_id)
    date_dir = os.path.join(user_dir, time.strftime("%Y%m%d"))

    for filename in glob.glob(os.path.join(date_dir, '*.png')):
        bucket = storage.bucket("hybpy-test.appspot.com")
        blob = bucket.blob(f'{user_id}/plots/{folder_id}/{os.path.basename(filename)}')
        blob.upload_from_filename(filename)
        blob.make_public()
        plot_urls.append(blob.public_url)
    
    all_blobs = list(bucket.list_blobs(prefix=f'{user_id}/plots/{folder_id}/'))
    seen_files = {}
    
    for blob in all_blobs:
        file_name = blob.name.split('/')[-1]
        base_name = re.sub(r'_[0-9]+\.png$', '', file_name)
        
        if base_name in seen_files:
            seen_files[base_name].append(blob)
        else:
            seen_files[base_name] = [blob]
    
    for base_name, blobs in seen_files.items():
        if len(blobs) > 1:
            # More than one file with the same base name, keep only one
            blobs_to_keep = [blobs[0]]
            blobs_to_delete = blobs[1:]
            
            for blob in blobs_to_delete:
                blob.delete()
    
    # Filter out duplicate URLs from plot_urls
    unique_plot_urls = []
    unique_base_names = set()
    
    for url in plot_urls:
        base_name = re.sub(r'_[0-9]+\.png$', '', os.path.basename(url))
        if base_name not in unique_base_names:
            unique_base_names.add(base_name)
            unique_plot_urls.append(url)

    return unique_plot_urls

def ensure_json_serializable(data):
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        return {k: ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_json_serializable(v) for v in data]
    else:
        return str(data)

def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Deleted directory: {directory_path}")
    except Exception as e:
        print(f"Error deleting directory {directory_path}: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        mode = request.form.get('mode')
        user_id = request.form.get('userId')
        description = request.form.get('description')
        train_batches = request.form.get('train_batches').split(",")
        test_batches = request.form.get('test_batches').split(",")
        user_id = request.form.get('user_id')

        if not file1 or not file2:
            return {"error": "Both files are required"}, 400
        
        logging.debug("Files received: file1=%s, file2=%s", file1.filename, file2.filename)

        file1.save(file1.filename)
        file2.save(file2.filename)

        folder_id = str(uuid.uuid4())

        file1_url = upload_file_to_storage(file1, user_id, file1.filename, folder_id)
        file2_url = upload_file_to_storage(file2, user_id, file2.filename, folder_id)

        logging.debug("Files uploaded: file1_url=%s, file2_url=%s", file1_url, file2_url)

        projhyb = hybdata(file1.filename)
        logging.debug("projhyb loaded: %s", projhyb)

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

        logging.debug("Data loaded from file2: %s", data)

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

        logging.debug("Data prepared for training: %s", data)

        projhyb, trainData = hybtrain(projhyb, data, user_id)

        logging.debug("Training complete: projhyb=%s, trainData=%s", projhyb, trainData)

        projhyb_serializable = ensure_json_serializable(projhyb)
        trainData_serializable = ensure_json_serializable(trainData)
        
        response_data = {
            "message": "Files processed successfully",
            "projhyb": projhyb_serializable,
            "trainData": trainData_serializable
        }

        plot_urls = upload_plots_to_gcs(user_id, folder_id)
        add_run(file1_url, file2_url, file1.filename, file2.filename, response_data, user_id, description, mode, plot_urls)

        # Clean up files from local storage
        #os.remove(file1.filename)
        #os.remove(file2.filename)
        delete_directory(os.path.join('plots'))


        return json.dumps(response_data), 200

    except Exception as e:
        logging.error("Error during file upload: %s", str(e), exc_info=True)
        return {"error": str(e)}, 500

@app.route('/add-run', methods=['POST'])
def add_run(file1_url, file2_url, file1, file2, response_data, user_id, description, mode, plot_urls):
    try:
        # Ensure `response_data` is serializable
        response_data_serializable = ensure_json_serializable(response_data)

        run_data = {
            "userId": user_id,
            "file1": file1_url,
            "file2": file2_url,
            "file1_name": file1,
            "file2_name": file2,
            "response_data": response_data_serializable,
            "description": description,
            "mode": mode,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "plots": plot_urls,
        }
        user_ref = db.collection('users').document(user_id)
        ren_ref = user_ref.collection('runs').document()
        ren_ref.set(run_data)
        return json.dumps(response_data), 200
        
    except Exception as e: 
        logging.error("Error in add_run: %s", str(e), exc_info=True)
        return {"error": str(e)}, 500
    


@app.route("/get-available-batches", methods=['POST'])
def get_available_batches():
    try:
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
            logging.debug("Available batches: %s", all_batches)

            return json.dumps(all_batches), 200
        
        else:
            return {"message": "File not found"}, 400
    except Exception as e:
        logging.error("Error in get_available_batches: %s", str(e), exc_info=True)
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)