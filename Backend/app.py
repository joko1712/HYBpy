from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
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
import tempfile
import glob
import re
import time
from threading import Thread
import threading
import traceback
import numpy as np
import torch
import sympy as sp
from multiprocessing import Process
from copy import deepcopy
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybtrain import hybtrain
from hybdata import hybdata
from csv2json import label_batches, add_state_and_time_to_data
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from dotenv import load_dotenv
from odesfun import projhyb_cache


load_dotenv()

cred = credentials.Certificate(os.path.join(os.path.dirname(
    __file__), 'hybpy-test-firebase-adminsdk-20qxj-fc73476cba.json'))
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("STORAGE_BUCKET_NAME")
})

db = firestore.client()

running_threads = {}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=[
    "https://www.hybpy.com", "https://hybpy.com", "http://localhost:3000", "http://localhost:4000"
])

CORS(app, origins="*", supports_credentials=True)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = request.headers.get(
        'Origin', '*')
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('message')
def handle_message(data):
    print('received message:', data)
    socketio.send('Message received: ' + data)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def upload_file_to_storage(file_path, user_id, filename, folder_id):
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))
    blob = bucket.blob(f"{user_id}/{folder_id}/{filename}")

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = 'application/octet-stream'

    blob.upload_from_filename(file_path, content_type=content_type)
    blob.make_public()
    return blob.public_url

def upload_plots_to_gcs(temp_dir, user_id, folder_id, plots_dir=None):
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    if plots_dir is None:
        plots_dir = os.path.join(temp_dir, "plots", user_id, folder_id)

    plot_urls = []
    if not os.path.exists(plots_dir):
        logging.warning(f"No plots directory found at: {plots_dir}")
        return plot_urls

    for filename in glob.glob(os.path.join(plots_dir, "*.png")):
        blob = bucket.blob(
            f"{user_id}/plots/{folder_id}/{os.path.basename(filename)}"
        )
        blob.upload_from_filename(filename, content_type="image/png")
        blob.make_public()
        plot_urls.append(blob.public_url)

    return plot_urls


def ensure_json_serializable(data, path="root"):
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        return {str(k): ensure_json_serializable(v, path=f"{path}.{k}") for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return [ensure_json_serializable(v, path=f"{path}[{i}]") for i, v in enumerate(data)]
    elif isinstance(data, np.ndarray):
        return ensure_json_serializable(data.tolist(), path=path)
    elif isinstance(data, np.generic):
        return ensure_json_serializable(data.item(), path=path)
    elif torch and isinstance(data, torch.Tensor):
        return ensure_json_serializable(data.detach().cpu().numpy().tolist(), path=path)
    else:
        print(f"❌ Non-serializable type at {path}: {type(data)} — {data}")
        raise TypeError(f"Cannot serialize type {type(data)} at path {path}")


def sanitize_projhyb(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_projhyb(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_projhyb(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, sp.Basic):
        return str(obj)
    elif callable(obj):
        return None
    elif hasattr(obj, '__module__') and 'torch' in obj.__module__:
        return None
    return obj


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"msg": "pong"}), 200


@app.route('/upload', methods=['OPTIONS', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'OPTIONS':
        return '', 204

    run_ref = None
    try:
        temp_dir = tempfile.mkdtemp()
        logging.debug("Form data received: %s", request.form)

        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        mode = request.form.get('mode')
        user_id = request.form.get('userId')
        description = request.form.get('description')
        train_batches = request.form.get('train_batches').split(
            ",") if request.form.get('train_batches') else []
        test_batches = request.form.get('test_batches').split(
            ",") if request.form.get('test_batches') else []
        val_batches = request.form.get('val_batches').split(
            ",") if request.form.get('val_batches') else []

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

        Crossval = request.form.get('Crossval')
        Ensemble = int(request.form.get('Ensemble'))
        Kfolds = int(request.form.get('Kfolds'))

        split_ratio_str = request.form.get("split_ratio", "").strip()
        try:
            split_ratio = float(split_ratio_str)
        except ValueError:
            split_ratio = 0.66

        trained_weights = request.form.get('trained_weights')
        logging.debug("trained_weights: %s", trained_weights)
        if trained_weights:
            trained_weights = trained_weights.strip(
                '[]"').replace('\n', '').replace(' ', '')
            trained_weights = list(map(float, trained_weights.split(',')))
        else:
            trained_weights = None

        if not file1 or not file2:
            return jsonify({"error": "Both files are required"}), 400

        execution_location = request.form.get("execution_location", "cloud")

        logging.debug("Files received: file1=%s, file2=%s",
                      file1.filename, file2.filename)

        # Save files to temporary directory
        file1_path = os.path.join(temp_dir, file1.filename)
        file2_path = os.path.join(temp_dir, file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)

        folder_id = str(uuid.uuid4())

        # Upload files to storage
        file1_url = upload_file_to_storage(
            file1_path, user_id, file1.filename, folder_id)
        file2_url = upload_file_to_storage(
            file2_path, user_id, file2.filename, folder_id)

        if not file1_url or not file2_url:
            return jsonify({"error": "Failed to upload files to storage"}), 500

        # Initialize hybdata with file1
        projhyb = hybdata(file1_path)

        Inputs = projhyb["inputs"]
        Outputs = projhyb["outputs"]

        user_ref = db.collection('users').document(user_id)
        run_ref = user_ref.collection('runs').document()

        use_validation = 'true' if Crossval == 1 else 'false'

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
                    "Bootstrap": Bootstrap,
                    "Crossval": Crossval,
                    "Ensemble": Ensemble,
                    "Kfolds": Kfolds
                },
                "status": "training in progress locally" if execution_location == "local" else "training in progress",
                "folder_id": folder_id,
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
                    "Bootstrap": Bootstrap,
                    "Crossval": Crossval,
                    "Ensemble": Ensemble,
                    "Kfolds": Kfolds
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

        plots_dir = os.path.join(temp_dir, 'plots', user_id, folder_id)
        os.makedirs(plots_dir, exist_ok=True)
        projhyb['plots_dir'] = plots_dir

        projhyb['kfolds'] = int(Kfolds)
        projhyb['nensemble'] = int(Ensemble)
        projhyb["test_batches"] = test_batches
        projhyb["crossval"] = Crossval
        projhyb["split_ratio"] = split_ratio

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
        '''
        
        projhyb, trainData, metrics, newHmodFile = hybtrain(projhyb, data, user_id, trained_weights, file1_path, temp_dir)

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

        plot_urls = upload_plots_to_gcs(temp_dir, user_id, folder_id)
        run_ref.update({
            "response_data": response_data,
            "status": "completed",
            "plots": plot_urls,
            "finishedAt": firestore.SERVER_TIMESTAMP,
        })

        shutil.rmtree(temp_dir)
    

        thread = Thread(
            target=background_training,
            args=(projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_ref, folder_id),
        )
         thread.do_run = True

        thread.start()
        running_threads[run_ref.id] = thread
        '''

        projhyb_for_local = sanitize_projhyb(deepcopy(projhyb))
        data_for_local = ensure_json_serializable(deepcopy(data))
        trained_weights_for_local = trained_weights
        if execution_location == "cloud":
            proc = Process(
                target=background_training,
                args=(projhyb, data, user_id, trained_weights,
                      file1_path, temp_dir, run_ref, folder_id),
            )

            proc.start()
            running_threads[run_ref.id] = proc

            return jsonify({"message": "Training started"}), 202
        else:
            return jsonify({
                "message": "Run created, awaiting local training",
                "run_id": run_ref.id,
                "user_id": user_id,
                "file1_url": file1_url,
                "file2_url": file2_url,
                "mode": mode,
                "train_batches": train_batches,
                "test_batches": test_batches,
                "val_batches": val_batches,
                "Crossval": Crossval,
                "Ensemble": Ensemble,
                "Kfolds": Kfolds,
                "split_ratio": split_ratio,
                "trained_weights": trained_weights_for_local,
                "folder_id": folder_id,
            }), 200
    except Exception as e:
        logging.error("Error in upload_file: %s", str(e), exc_info=True)
        if run_ref:
            run_ref.update({"status": "error"})
        shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 500


def background_training(projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_ref, folder_id):
    t = threading.current_thread()
    run_id = run_ref.id
    try:
        projhyb["run_id"] = run_id

        projhyb, trainData, metrics, newHmodFile = hybtrain(
            projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_id=run_id, thread=t
        )

        if not getattr(t, "do_run", True):
            logging.info(f"Training stopped by user: {run_id}")
            run_ref.update({"status": "cancelled"})
            return

        new_hmod_filename = os.path.basename(newHmodFile)
        new_hmod_url = upload_file_to_storage(
            newHmodFile, user_id, new_hmod_filename, folder_id)

        response_data = {
            "message": "Files processed successfully",
            # "projhyb": sanitize_projhyb(projhyb),
            # "trainData": ensure_json_serializable(trainData),
            "trainData": json.dumps(ensure_json_serializable(trainData), default=str),
            "metrics": metrics,
            "new_hmod_url": new_hmod_url,
            "new_hmod": new_hmod_filename,
        }

        plot_urls = upload_plots_to_gcs(
            temp_dir, user_id, folder_id, projhyb.get('plots_dir'))

        run_ref.update({
            "response_data": response_data,
            "status": "completed",
            "plots": plot_urls,
            "finishedAt": firestore.SERVER_TIMESTAMP,
        })

        try:
            user_email = get_user_email(user_id)

            email_data = {
                "user_email": user_email,
                "user_id": user_id,
                "run_id": run_id,
                "run_name": run_ref.get().to_dict().get("description", "No description provided"),
                "status": "completed",
                "metrics": metrics,
                "new_hmod_url": new_hmod_url,
                "new_hmod_filename": new_hmod_filename,
            }
            email_function_url = 'https://us-central1-hybpy-test.cloudfunctions.net/send_email_notification'
            email_response = requests.post(email_function_url, json=email_data)
            email_response.raise_for_status()
        except Exception as e:
            logging.error("Email notification failed: %s",
                          str(e), exc_info=True)
            if run_id in projhyb_cache:
                projhyb_cache.pop(run_id, None)

    except Exception as e:
        logging.error("Error during background training: %s",
                      str(e), exc_info=True)
        run_ref.update({"status": "error"})

    finally:
        running_threads.pop(run_id, None)
        shutil.rmtree(temp_dir)


@app.route("/upload-local-results", methods=["POST"])
def upload_local_results():
    user_id = request.form["user_id"]
    run_id = request.form["run_id"]
    metrics_str = request.form.get("metrics", "{}")
    new_hmod_file = request.files.get("file")
    train_data_str = request.form.get("train_data", "{}")

    run_ref = db.collection("users").document(
        user_id).collection("runs").document(run_id)

    run_data = run_ref.get().to_dict()
    folder_id = run_data.get("folder_id")

    # save temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, new_hmod_file.filename)
    new_hmod_file.save(tmp_path)

    new_hmod_url = upload_file_to_storage(
        tmp_path, user_id, new_hmod_file.filename, folder_id
    )

    try:
        metrics = json.loads(metrics_str or "{}")
    except Exception:
        metrics = {}

    response_data = {
        "message": "Files processed successfully",
        "trainData": train_data_str,
        "metrics": metrics,
        "new_hmod_url": new_hmod_url,
        "new_hmod": new_hmod_file.filename,
    }


    run_ref.update({
        "status": "completed",
        "finishedAt": firestore.SERVER_TIMESTAMP,
        "metrics": metrics,
        "new_hmod_url": new_hmod_url,
        "new_hmod": new_hmod_file.filename,
        "response_data": response_data,
    })

    return jsonify({"message": "Local results stored"}), 200

@app.route("/upload-local-plots", methods=["POST"])
def upload_local_plots():
    try:
        user_id = request.form["user_id"]
        run_id = request.form["run_id"]
        plots_zip = request.files.get("plots_zip")

        if not plots_zip:
            return jsonify({"error": "Missing plots_zip file"}), 400

        run_ref = db.collection("users").document(
            user_id
        ).collection("runs").document(run_id)
        run_data = run_ref.get().to_dict()

        if not run_data:
            return jsonify({"error": "Run not found"}), 404

        folder_id = run_data.get("folder_id")
        if not folder_id:
            return jsonify({"error": "Missing folder_id on run"}), 400

        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, plots_zip.filename)
        plots_zip.save(zip_path)

        extract_dir = os.path.join(tmp_dir, "plots")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        plot_urls = upload_plots_to_gcs(
            temp_dir=tmp_dir,
            user_id=user_id,
            folder_id=folder_id,
            plots_dir=extract_dir,
        )

        run_ref.update({"plots": plot_urls})

        shutil.rmtree(tmp_dir, ignore_errors=True)

        return jsonify({"message": "Local plots stored", "count": len(plot_urls)}), 200

    except Exception as e:
        logging.error("Error in upload-local-plots: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


def get_user_email(uid):
    try:
        user = auth.get_user(uid)
        return user.email
    except Exception as e:
        logging.error(f"Failed to fetch email for UID {uid}: {e}")
        return None


@app.route("/cancel-run", methods=["POST"])
@cross_origin()
def cancel_run():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        runs_ref = db.collection('users').document(user_id).collection('runs')
        latest_run = runs_ref.order_by(
            "createdAt", direction=firestore.Query.DESCENDING).limit(1).get()

        if not latest_run:
            return jsonify({"error": "No active run found"}), 404

        run_doc = latest_run[0]
        run_id = run_doc.id

        runs_ref.document(run_id).update({"cancel_requested": True})

        print("doc", runs_ref.document(run_id).get().to_dict())

        if run_id in running_threads:
            running_threads[run_id].do_run = False
            return jsonify({"message": "Run cancellation requested (thread + flag)"}), 200
        else:
            return jsonify({"message": "Run cancellation requested (flag only)"}), 200

    except Exception as e:
        logging.error(f"Error in cancel_run: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/get-available-batches", methods=['OPTIONS', 'POST'])
@cross_origin()
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
        logging.error("Error in get_available_batches: %s",
                      str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/run-status', methods=['OPTIONS', 'GET'])
@cross_origin()
def run_status():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    try:
        user_ref = db.collection('users').document(user_id)
        run_ref = user_ref.collection('runs')\
            .order_by('createdAt', direction=firestore.Query.DESCENDING)\
            .limit(1)

        latest_run = list(run_ref.stream())

        if latest_run:
            run_data = latest_run[0].to_dict()
            status = run_data.get('status', 'unknown')
            return jsonify({"status": status}), 200

        return jsonify({"status": "no_runs"}), 200
    except Exception as e:
        logging.error("Error in run_status: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/get-template-hmod', methods=['OPTIONS', 'POST'])
@cross_origin()
def get_template_hmod():

    if request.method == 'OPTIONS':
        return '', 204

    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 1:
            blob_hmod = bucket.blob(
                "Template/CellGrowthModel/CellGrowthModel3.hmod")
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
        print("Error in /get-template-hmod:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/get-template-csv', methods=['OPTIONS', 'POST'])
@cross_origin()
def get_template_csv():
    if request.method == 'OPTIONS':
        return '', 204

    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 1:
            blob_csv = bucket.blob(
                "Template/CellGrowthModelCSV/CellGrowth.csv")
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
        print("Error in /get-template-csv:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/get-template-xlsx', methods=['OPTIONS', 'POST'])
@cross_origin()
def get_template_xlsx():
    bucket = storage.bucket(os.getenv("STORAGE_BUCKET_NAME"))

    try:
        template_type = request.json.get('template_type')
        if template_type == 3:
            blob_xlsx = bucket.blob(
                "Template/Csv/template_datafile_hybpy.xlsx")
            xlsx_file_path = "template_datafile_hybpy.xlsx"
        else:
            return jsonify({"error": "Invalid template type"}), 400

        temp_file = NamedTemporaryFile(delete=False)
        blob_xlsx.download_to_filename(temp_file.name)

        return send_file(temp_file.name, as_attachment=True, download_name=xlsx_file_path)

    except Exception as e:
        logging.error("Error in get_template_xlsx: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/get-template-hmod-download', methods=['OPTIONS', 'POST'])
@cross_origin()
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


@app.route('/delete-run', methods=['OPTIONS', 'DELETE'])
@cross_origin()
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
                logging.error(
                    f"Error deleting blob: {blob_path}, error: {str(e)}")

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


@app.route('/get-file-urls', methods=['OPTIONS', 'GET'])
@cross_origin()
def get_file_urls():
    user_id = request.args.get('user_id')
    run_id = request.args.get('run_id')

    try:
        run_ref = db.collection('users').document(
            user_id).collection('runs').document(run_id)
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


@app.route("/get-new-hmod", methods=['OPTIONS', 'POST'])
@cross_origin()
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
    socketio.run(app, host="0.0.0.0", port=5000, ssl_context=(
        '/etc/letsencrypt/live/api.hybpy.com/fullchain.pem',
        '/etc/letsencrypt/live/api.hybpy.com/privkey.pem'
    ))
