from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os
import json
import csv
import logging
import uuid
import shutil
import mimetypes
import requests
from tempfile import NamedTemporaryFile
import tempfile  # For creating temporary directories
import glob
import re
import time
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom modules (ensure these are thread-safe)
from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, add_state_and_time_to_data


# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Firebase Admin SDK
#cred = credentials.Certificate("../hybpy-test-firebase-adminsdk-20qxj-ebfca8f109.json")
cred = credentials.Certificate("../hybpy-test-firebase-adminsdk-20qxj-245fd03d89.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("STORAGE_BUCKET_NAME")
})

db = firestore.client()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

def upload_file_to_storage(file_path, user_id, filename, folder_id):
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))
    blob = bucket.blob(f"{user_id}/{folder_id}/{filename}")

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = 'application/octet-stream'

    blob.upload_from_filename(file_path, content_type=content_type)
    blob.make_public()
    return blob.public_url

def upload_plots_to_gcs(temp_dir, user_id, folder_id):
    plot_urls = []
    temp = os.path.join(temp_dir, 'plots')
    user_dir = os.path.join(temp, user_id)
    date_dir = os.path.join(user_dir, time.strftime("%Y%m%d"))

    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

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

def ensure_json_serializable(data):
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        return {k: ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_json_serializable(v) for v in data]
    else:
        return str(data)

@app.route('/upload', methods=['POST'])
def upload_file():
    run_ref = None
    try:
        temp_dir = tempfile.mkdtemp()  # Create a unique temporary directory
        logging.debug("Form data received: %s", request.form)

        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        mode = request.form.get('mode')
        user_id = request.form.get('userId')
        description = request.form.get('description')
        train_batches = request.form.get('train_batches').split(",") if request.form.get('train_batches') else []
        test_batches = request.form.get('test_batches').split(",") if request.form.get('test_batches') else []

        HiddenNodes = request.form.get('HiddenNodes')
        Layer = request.form.get('Layer')
        Tau = request.form.get('Tau')
        Mode = request.form.get('Mode')
        Method = request.form.get('Method')
        Jacobian = request.form.get('Jacobian')
        Hessian = request.form.get('Hessian')
        Niter = request.form.get('Niter')
        Nstep = request.form.get('Nstep')
        Bootstrap = request.form.get('Bootstrap')
        Inputs = request.form.get('Inputs')
        Outputs = request.form.get('Outputs')

        trained_weights = request.form.get('trained_weights')
        logging.debug("trained_weights: %s", trained_weights)
        if trained_weights:
            trained_weights = trained_weights.strip('[]"').replace('\n', '').replace(' ', '')
            trained_weights = list(map(float, trained_weights.split(',')))
        else:
            trained_weights = None

        if not file1 or not file2:
            return jsonify({"error": "Both files are required"}), 400

        logging.debug("Files received: file1=%s, file2=%s", file1.filename, file2.filename)

        # Save files to temporary directory
        file1_path = os.path.join(temp_dir, file1.filename)
        file2_path = os.path.join(temp_dir, file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)

        folder_id = str(uuid.uuid4())

        # Upload files to storage
        file1_url = upload_file_to_storage(file1_path, user_id, file1.filename, folder_id)
        file2_url = upload_file_to_storage(file2_path, user_id, file2.filename, folder_id)

        if not file1_url or not file2_url:
            return jsonify({"error": "Failed to upload files to storage"}), 500

        # Initialize hybdata with file1
        projhyb = hybdata(file1_path)

        Inputs = projhyb["inputs"]
        Outputs = projhyb["outputs"]

        user_ref = db.collection('users').document(user_id)
        run_ref = user_ref.collection('runs').document()

        if trained_weights == None:
            run_ref.set({
                "userId": user_id,
                "file1": file1_url,
                "file2": file2_url,
                "file1_name": file1.filename,
                "file2_name": file2.filename,
                "description": description,
                "trained_weights": trained_weights,
                "mode": mode,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "Inputs": Inputs,
                "Outputs": Outputs,
                "MachineLearning": {
                    "HiddenNodes": HiddenNodes,
                    "Layer": Layer,
                    "Tau": Tau,
                    "Mode": Mode,
                    "Method": Method,
                    "Jacobian": Jacobian,
                    "Hessian": Hessian,
                    "Niter": Niter,
                    "Nstep": Nstep,
                    "Bootstrap": Bootstrap
                },
                "status": "training in progress",
            })
        else:
            run_ref.set({
                "userId": user_id,
                "file1": file1_url,
                "file2": file2_url,
                "file1_name": file1.filename,
                "file2_name": file2.filename,
                "description": description,
                "trained_weights": trained_weights,
                "mode": mode,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "Inputs": Inputs,
                "Outputs": Outputs,
                "MachineLearning": {
                    "HiddenNodes": HiddenNodes,
                    "Layer": Layer,
                    "Tau": Tau,
                    "Mode": Mode,
                    "Method": Method,
                    "Jacobian": Jacobian,
                    "Hessian": Hessian,
                    "Niter": Niter,
                    "Nstep": Nstep,
                    "Bootstrap": Bootstrap
                },
                "status": "simulation in progress",
            })

        # Process file2 and prepare data
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

        data = add_state_and_time_to_data(data, projhyb)
        count = len(data)
        data["nbatch"] = count


        data_json_path = os.path.join(temp_dir, "data.json")
        with open(data_json_path, "w") as write_file:
            json.dump(data, write_file)

        projhyb_json_path = os.path.join(temp_dir, "projhyb.json")
        with open(projhyb_json_path, "w") as write_file:
            json.dump(projhyb, write_file)

        '''
        files = {
            'projhyb_file': open(projhyb_json_path, 'rb'),
            'data_file': open(data_json_path, 'rb')
        }

        print("trained_weights", trained_weights)

        data_params = {
            'user_id': user_id,
            'trained_weights': json.dumps(trained_weights) if trained_weights else None,
            'file1_url': file1_url,
            'file1Filename': file1.filename,
            "folder_id": folder_id,
            "bucket_name": os.getenv("STORAGE_BUCKET_NAME"),
            "run_ref_id": run_ref.id,
        }
        
        #gcloud functions deploy run_hybtrain --runtime python39 --trigger-http --allow-unauthenticated --project hybpy-test --region us-central1 --entry-point run_hybtrain --memory 2048MB --timeout 3600s

        cloud_function_url = 'https://us-central1-hybpy-test.cloudfunctions.net/run_hybtrain'

        response = requests.post(cloud_function_url, data=data_params, files=files)
        response.raise_for_status()
        '''

        
        # Adjust hybtrain to accept temp_dir and use paths accordingly
        projhyb, trainData, metrics, newHmodFile = hybtrain(projhyb, data, user_id, trained_weights, file1_path, temp_dir)

        # Upload new Hmod file to storage
        new_hmod_filename = os.path.basename(newHmodFile)
        new_hmod_url = upload_file_to_storage(newHmodFile, user_id, new_hmod_filename, folder_id)

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

        # Upload plots to storage
        plot_urls = upload_plots_to_gcs(temp_dir, user_id, folder_id)
        run_ref.update({
            "response_data": response_data,
            "status": "completed",
            "plots": plot_urls,
            "finishedAt": firestore.SERVER_TIMESTAMP,
        })

        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        

        return jsonify("Done") , 200

    except Exception as e:
        logging.error("Error during file upload: %s", str(e), exc_info=True)
        
        run_ref.update({"status": "error"})
        return jsonify({"error": str(e)}), 500
        

@app.route("/get-available-batches", methods=['POST'])
def get_available_batches():
    try:
        temp_dir = tempfile.mkdtemp()
        file2 = request.files.get('file2')

        if file2:
            file2_path = os.path.join(temp_dir, file2.filename)
            file2.save(file2_path)

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

            all_batches = list(data.keys())

            shutil.rmtree(temp_dir)
            return jsonify(all_batches), 200

        else:
            shutil.rmtree(temp_dir)
            return jsonify({"message": "File not found"}), 400
    except Exception as e:
        logging.error("Error in get_available_batches: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/run-status', methods=['GET'])
def run_status():
    user_id = request.args.get('user_id')
    try:
        user_ref = db.collection('users').document(user_id)
        run_ref = user_ref.collection('runs').order_by('createdAt', direction=firestore.Query.DESCENDING).limit(1)
        latest_run = list(run_ref.stream())

        if latest_run:
            run_data = latest_run[0].to_dict()
            status = run_data.get('status', 'unknown')
            return jsonify({"status": status}), 200

        return jsonify({"status": "no_runs"}), 200
    except Exception as e:
        logging.error("Error in run_status: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-template-hmod', methods=['POST'])
def get_template_hmod():
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 1:
            blob_hmod = bucket.blob("Template/CellGrowthModel/CellGrowthModel3.hmod")
            hmod_file_path = "CellGrowthModel3.hmod"
        elif template_type == 2:
            blob_hmod = bucket.blob("Template/Hmod1/basic1.hmod")
            hmod_file_path = "basic1.hmod"
        else:
            return jsonify({"error": "Invalid template type"}), 400

        temp_file = NamedTemporaryFile(delete=False)
        blob_hmod.download_to_filename(temp_file.name)

        return send_file(temp_file.name, as_attachment=True, download_name=hmod_file_path)

    except Exception as e:
        logging.error("Error in get_template_hmod: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-template-csv', methods=['POST'])
def get_template_csv():
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 1:
            blob_csv = bucket.blob("Template/CellGrowthModelCSV/CellGrowth.csv")
            csv_file_path = "CellGrowth.csv"
        elif template_type == 2:
            blob_csv = bucket.blob("Template/Csv/basicmodel1data.csv")
            csv_file_path = "basicmodel1data.csv"
        else:
            return jsonify({"error": "Invalid template type"}), 400

        temp_file = NamedTemporaryFile(delete=False)
        blob_csv.download_to_filename(temp_file.name)

        return send_file(temp_file.name, as_attachment=True, download_name=csv_file_path)

    except Exception as e:
        logging.error("Error in get_template_csv: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-template-xlsx', methods=['POST'])
def get_template_xlsx():
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 3:
            blob_xlsx = bucket.blob("Template/Csv/template_datafile_hybpy.xlsx")
            xlsx_file_path = "template_datafile_hybpy.xlsx"
        else:
            return jsonify({"error": "Invalid template type"}), 400

        temp_file = NamedTemporaryFile(delete=False)
        blob_xlsx.download_to_filename(temp_file.name)

        return send_file(temp_file.name, as_attachment=True, download_name=xlsx_file_path)

    except Exception as e:
        logging.error("Error in get_template_xlsx: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-template-hmod-download', methods=['POST'])
def get_template_hmod_download():
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 3:
            blob_hmod = bucket.blob("Template/Hmod/template.hmod")
            hmod_file_path = "template.hmod"
        else:
            return jsonify({"error": "Invalid template type"}), 400

        temp_file = NamedTemporaryFile(delete=False)
        blob_hmod.download_to_filename(temp_file.name)

        return send_file(temp_file.name, as_attachment=True, download_name=hmod_file_path)

    except Exception as e:
        logging.error("Error in get_template_xlsx: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/delete-run', methods=['DELETE'])
def delete_run():
    try:
        data = request.json
        user_id = data.get('user_id')
        run_id = data.get('run_id')
        folder_path = data.get('folder_path')

        if not user_id or not run_id or not folder_path:
            return jsonify({"error": "User ID, Run ID, and Folder Path are required"}), 400

        user_ref = db.collection('users').document(user_id)
        run_ref = user_ref.collection('runs').document(run_id)
        run_data = run_ref.get().to_dict()

        if not run_data:
            return jsonify({"error": "Run not found"}), 404

        bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

        def delete_blob(blob_path):
            try:
                blob = bucket.blob(blob_path)
                blob.delete()
            except Exception as e:
                logging.error(f"Error deleting blob: {blob_path}, error: {str(e)}")

        folder_prefix = folder_path + '/'
        blobs = bucket.list_blobs(prefix=folder_prefix)
        for blob in blobs:
            delete_blob(blob.name)

        plots_folder_prefix = f"{user_id}/plots/{folder_path.split('/')[-1]}/"
        blobs = bucket.list_blobs(prefix=plots_folder_prefix)
        for blob in blobs:
            delete_blob(blob.name)

        run_ref.delete()
        logging.debug(f"Deleted Firestore document for run_id={run_id}")

        return jsonify({"message": "Run and associated files deleted successfully"}), 200

    except Exception as e:
        logging.error("Error deleting run: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-file-urls', methods=['GET'])
def get_file_urls():
    user_id = request.args.get('user_id')
    run_id = request.args.get('run_id')

    try:
        run_ref = db.collection('users').document(user_id).collection('runs').document(run_id)
        run_data = run_ref.get().to_dict()

        if not run_data:
            return jsonify({"error": "Run not found"}), 404

        file1_url = run_data.get('file1')
        file2_url = run_data.get('file2')
        new_hmod_url = run_data.get('response_data', {}).get('new_hmod_url')

        return jsonify({
            "file1_url": file1_url,
            "file2_url": file2_url,
            "new_hmod_url": new_hmod_url
        }), 200
    except Exception as e:
        logging.error("Error in get_file_urls: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/get-new-hmod", methods=['POST'])
def get_new_hmod():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        response = requests.get(url)
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(response.content)
        temp_file.close()

        return send_file(temp_file.name, as_attachment=True)
    except Exception as e:
        logging.error("Error in get_new_hmod: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=8080)