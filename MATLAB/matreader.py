from scipy.io import loadmat
import numpy as np
import json
import base64


def convert_mat_struct(obj):
    """
    Convert MATLAB structures to Python dictionaries recursively.
    """
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:  # It's a MATLAB struct
        result = {}
        for name in obj.dtype.names:
            element = obj[name].squeeze()
            result[name] = convert_mat_struct(element)
        return result
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    else:
        return obj


def load_and_process_mat_file(file_path):
    """
    Load a MATLAB .mat file and convert it to a Python dictionary.
    """
    mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    processed_data = {}
    for key in mat_data:
        if not key.startswith('__'):
            processed_data[key] = convert_mat_struct(mat_data[key])
    return processed_data


# Path to the MATLAB file
file_path = "/Users/joko1712/Documents/TESE/MATLAB/projhyb(Leeram).mat"

# Convert the MATLAB data
converted_data = load_and_process_mat_file(file_path)

# Write the converted data to a JSON file
with open('converted_data.json', 'w') as json_file:
    # Use a custom JSON encoder to handle any remaining unserializable data
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode('utf-8')
            return json.JSONEncoder.default(self, obj)

    json.dump(converted_data, json_file, indent=4, cls=CustomEncoder)

print("Data conversion and saving complete.")
