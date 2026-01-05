import os
import tempfile
import threading
import json
import csv
import shutil
import requests
import zipfile
import numpy as np
import torch
import time
import uuid
import zipfile


# BUILD:
'''
pyinstaller \
  --onefile local_trainer.py \
  --paths .. \
  --hidden-import hybdata \
  --hidden-import hybtrain
'''

from flask import Flask, request, jsonify
from flask_cors import CORS

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from csv2json import add_state_and_time_to_data
from csv2json import label_batches, add_state_and_time_to_data

from hybtrain import hybtrain
from hybdata import hybdata



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def download_to_file(url, suffix=""):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


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



def run_training(
    run_id,
    user_id,
    backend_url,
    file1_url,
    file2_url,
    mode,
    train_batches,
    test_batches,
    val_batches,
    Crossval,
    Ensemble,
    Kfolds,
    split_ratio,
    trained_weights,
    folder_id,
):
    try:
        backend_url = backend_url.rstrip("/")

        temp_dir = tempfile.mkdtemp()
        file1_path = download_to_file(file1_url, suffix=".hmod")
        file2_path = download_to_file(file2_url, suffix=".csv")
        print("Downloaded HMOD to:", file1_path)
        print("Downloaded CSV to:", file2_path)

        projhyb = hybdata(file1_path)
        projhyb["run_id"] = run_id

        data = {}
        current_time_group = 1
        current_time = None

        with open(file2_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

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
                if (
                    int(batch) not in train_batches
                    and int(batch) not in test_batches
                    and int(batch) not in val_batches
                ):
                    data[int(batch)]["istrain"] = [0]

        if mode in ("2", "3"):
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
                manual_test_batches=test_batches,
            )


        data = add_state_and_time_to_data(data, projhyb)
        count = len(data)
        data["nbatch"] = count

        plots_dir = os.path.join(temp_dir, "plots", user_id, folder_id or run_id)
        os.makedirs(plots_dir, exist_ok=True)
        projhyb["plots_dir"] = plots_dir

        if Kfolds is not None:
            projhyb["kfolds"] = int(Kfolds)
        if Ensemble is not None:
            projhyb["nensemble"] = int(Ensemble)
        projhyb["test_batches"] = test_batches
        projhyb["crossval"] = Crossval
        projhyb["split_ratio"] = split_ratio

        print("Starting local hybtrain...")
        projhyb, trainData, metrics, newHmodFile = hybtrain(
            projhyb,
            data,
            user_id,
            trained_weights,
            file1_path,
            temp_dir,
            run_id=run_id,
            thread=None,
        )
        print("Local hybtrain finished. New HMOD:", newHmodFile)

        with open(newHmodFile, "rb") as f:
            files = {
                "file": (os.path.basename(newHmodFile), f, "application/octet-stream")
            }
            payload = {
                "user_id": user_id,
                "run_id": run_id,
                "metrics": json.dumps(metrics),
                "train_data": json.dumps(
                    ensure_json_serializable(trainData), default=str
                ),
            }
            resp2 = requests.post(
                f"{backend_url}/upload-local-results",
                data=payload,
                files=files,
                timeout=180,
            )
            print("upload-local-results status:", resp2.status_code, resp2.text)
            resp2.raise_for_status()

        plots_zip_path = os.path.join(temp_dir, "plots.zip")
        with zipfile.ZipFile(plots_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(plots_dir):
                for fname in files:
                    if fname.lower().endswith(".png"):
                        full_path = os.path.join(root, fname)
                        rel_name = os.path.relpath(full_path, plots_dir)
                        zf.write(full_path, rel_name)

        with open(plots_zip_path, "rb") as fz:
            files = {"plots_zip": ("plots.zip", fz, "application/zip")}
            payload = {"user_id": user_id, "run_id": run_id}
            resp3 = requests.post(
                f"{backend_url}/upload-local-plots",
                data=payload,
                files=files,
                timeout=180,
            )
            print("upload-local-plots status:", resp3.status_code, resp3.text)
            resp3.raise_for_status()

    except Exception as e:
        print("Local trainer error:", e)
        try:
            requests.post(
                f"{backend_url}/local-run-error",
                json={"user_id": user_id, "run_id": run_id, "error": str(e)},
                timeout=10,
            )
        except Exception:
            pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/train", methods=["POST", "OPTIONS"])
def train():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    print("Received /train payload keys:", list(data.keys()))

    run_id = data.get("run_id") or data.get("runId")
    user_id = data.get("user_id") or data.get("userId")
    backend_url = data.get("backend_url", "https://api.hybpy.com")

    file1_url = data.get("file1_url")
    file2_url = data.get("file2_url")
    mode = data.get("mode")
    train_batches = data.get("train_batches") or []
    test_batches = data.get("test_batches") or []
    val_batches = data.get("val_batches") or []
    Crossval = data.get("Crossval")
    Ensemble = data.get("Ensemble")
    Kfolds = data.get("Kfolds")
    split_ratio = data.get("split_ratio")
    trained_weights = data.get("trained_weights")
    folder_id = data.get("folder_id")

    if not run_id or not user_id or not file1_url or not file2_url:
        return jsonify({
            "error": "Missing required fields",
            "received": list(data.keys()),
        }), 400

    t = threading.Thread(
        target=run_training,
        args=(
            run_id,
            user_id,
            backend_url,
            file1_url,
            file2_url,
            mode,
            train_batches,
            test_batches,
            val_batches,
            Crossval,
            Ensemble,
            Kfolds,
            split_ratio,
            trained_weights,
            folder_id,
        ),
        daemon=True,
    )
    t.start()

    return jsonify({"status": "started", "run_id": run_id})

@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"status": "ok", "source": "local_trainer"}), 200


def zip_dir(src_dir):
    fd, zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, src_dir)
                zf.write(full_path, rel_path)
    return zip_path


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4000)
