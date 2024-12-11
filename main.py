import json
import tempfile
import os
import uuid
import shutil
import logging
from flask import jsonify, request
import firebase_admin
from firebase_admin import credentials, firestore, storage
import mimetypes
import time
import glob
import re


# Ensure logging is configured
logging.basicConfig(level=logging.DEBUG)

# Import your custom modules
from hybtrain import hybtrain  # Ensure this module is included
# Include other modules as needed

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "hybpy-test-firebase-adminsdk-20qxj-245fd03d89.json")

cred = credentials.Certificate(json_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': "hybpy-test.appspot.com"
})

# Initialize Cloud Storage client
db = firestore.client()

import functions_framework
@functions_framework.http
def run_hybtrain(request):
    try:
        logging.debug("Content-Type: %s", request.content_type)

        # Check if the request contains multipart/form-data
        if not request.content_type.startswith('multipart/form-data'):
            logging.error("Invalid Content-Type")
            return jsonify({"error": "Content-Type must be multipart/form-data"}), 400

        # Get form fields
        user_id = request.form.get('user_id')
        trained_weights_json = request.form.get('trained_weights')
        file1_url = request.form.get('file1_url')
        file1_filename = request.form.get('file1Filename')
        folder_id = request.form.get('folder_id')
        bucket_name = request.form.get('bucket_name')
        run_ref_id = request.form.get('run_ref_id')

        logging.debug("Form Data: user_id=%s, file1_url=%s, file1_filename=%s", user_id, file1_url, file1_filename)
        logging.debug("Form Data: folder_id=%s, bucket_name=%s, run_ref_id=%s", folder_id, bucket_name, run_ref_id)

        # Convert trained_weights back to a Python object
        trained_weights = json.loads(trained_weights_json) if trained_weights_json else None

        logging.debug("Trained Weights: %s", trained_weights)

        # Handle uploaded files
        projhyb_file = request.files.get('projhyb_file')
        data_file = request.files.get('data_file')

        if not projhyb_file or not data_file:
            logging.error("Missing uploaded files")
            return jsonify({"error": "projhyb_file and data_file are required"}), 400

        logging.debug("Received files: projhyb_file=%s, data_file=%s", projhyb_file.filename, data_file.filename)

        # Save the files to temporary paths
        temp_dir = tempfile.mkdtemp()
        logging.debug("Temporary directory created at %s", temp_dir)

        projhyb_path = os.path.join(temp_dir, 'projhyb.json')
        data_path = os.path.join(temp_dir, 'data.json')

        projhyb_file.save(projhyb_path)
        data_file.save(data_path)

        logging.debug("Files saved to temporary directory")

        # Load the data from the files
        with open(projhyb_path, 'r') as f:
            projhyb = json.load(f)
        logging.debug("Loaded projhyb data")

        with open(data_path, 'r') as f:
            data = json.load(f)
        logging.debug("Loaded data")


        # Download file1
        file1_path = os.path.join(temp_dir, file1_filename)
        download_file_from_url(file1_url, file1_path)
        logging.debug("Downloaded file1 to %s", file1_path)

        # Run hybtrain
        logging.debug("Running hybtrain function")
        projhyb_result, trainData, metrics, newHmodFile = hybtrain(
            projhyb, data, user_id, trained_weights, file1_path, temp_dir
        )
        logging.debug("hybtrain function completed successfully")

        # Upload newHmodFile to Cloud Storage
        new_hmod_filename = os.path.basename(newHmodFile)
        new_hmod_url = upload_file_to_storage(newHmodFile, user_id, new_hmod_filename, folder_id, bucket_name)


        projhyb_serializable = ensure_json_serializable(projhyb)
        trainData_serializable = ensure_json_serializable(trainData)

        response_data = {
            "message": "Files processed successfully",
            "projhyb": projhyb_serializable,
            "trainData": trainData_serializable,
            "metrics": metrics,
            "new_hmod_url": new_hmod_url,
            "new_hmod": new_hmod_filename,
        }

        plot_urls = upload_plots_to_gcs(temp_dir, user_id, folder_id, bucket_name)

        user_ref = db.collection('users').document(user_id)
        run_ref = user_ref.collection('runs').document(run_ref_id)
        run_ref.update({
            "response_data": response_data,
            "status": "completed",
            "plots": plot_urls,
            "finishedAt": firestore.SERVER_TIMESTAMP,
        })
        logging.debug("Firestore run_ref updated successfully")
        

        shutil.rmtree(temp_dir)
        logging.debug("Temporary directory cleaned up")

        return jsonify("Done"), 200

    except Exception as e:
        logging.exception("Exception occurred in run_hybtrain")
        return json.dumps({'error': str(e)}), 500

def download_file_from_url(url, destination):
    import requests
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)


def upload_file_to_storage(file_path, user_id, filename, folder_id, bucket_name):
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(f"{user_id}/{folder_id}/{filename}")

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = 'application/octet-stream'

    blob.upload_from_filename(file_path, content_type=content_type)
    blob.make_public()
    return blob.public_url

def ensure_json_serializable(data):
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        return {k: ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_json_serializable(v) for v in data]
    else:
        return str(data)


def upload_plots_to_gcs(temp_dir, user_id, folder_id, bucket_name):
    plot_urls = []
    temp = os.path.join(temp_dir, 'plots')
    user_dir = os.path.join(temp, user_id)
    date_dir = os.path.join(user_dir, time.strftime("%Y%m%d"))

    bucket = storage.bucket(bucket_name)

    for filename in glob.glob(os.path.join(date_dir, '*.png')):
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
            blobs_to_delete = blobs[1:]  # Keep the first one, delete the rest
            
            for blob in blobs_to_delete:
                try:
                    blob.delete()
                except Exception as e:
                    logging.error(f"Error deleting blob: {blob.name}, error: {str(e)}")

    unique_plot_urls = [blob.public_url for blob_list in seen_files.values() for blob in blob_list]

    return unique_plot_urls