import base64
import os
import pickle
import time

import numpy as np
import pandas as pd


def to_csv(data, save_dir: str, save_name: str):
    """
    Save the input data (either a DataFrame or a 3D NumPy array) into CSV file(s).

    If the input is a pandas DataFrame, it will be saved directly to the given directory.
    If the input is a 3D NumPy array (with shape [num, time, dim]), each 2D slice (data[i])
    will be saved into a separate subdirectory named 'sample_i'.

    :param data: The data to save, either a pandas DataFrame or a NumPy array of shape (num, time, dim).
    :param save_dir: The root directory where the files will be saved.
    :param save_name: The name of the CSV file(s) to be written.
    :raises TypeError: If the input data type is not supported.
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(os.path.join(save_dir, save_name), index=False)

    elif isinstance(data, np.ndarray):
        num = data.shape[0]
        for i in range(num):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            df = pd.DataFrame(data[i])
            df.to_csv(os.path.join(sample_dir, save_name), index=False)

    elif isinstance(data, list):
        num = len(data)
        for i in range(num):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            df = data[i]
            df.to_csv(os.path.join(sample_dir, save_name), index=False)

    else:
        raise TypeError("Unsupported type for data. Must be pd.DataFrame or np.ndarray.")


def decode_data(filepath: str):
    """
    Load the result CSV file and decode the base64-encoded 'inference_data' and 'actual_data' columns.

    :param filepath: Path to the input CSV file containing encoded data.
    :return: None. The decoded data will be saved as CSV files in corresponding folders.
    """
    data = pd.read_csv(filepath)  # Read the CSV file with encoded columns

    for index, row in data.iterrows():
        # Decode base64 strings and deserialize them back to original DataFrames
        decoded_inference_data = base64.b64decode(row["inference_data"])
        decoded_actual_data = base64.b64decode(row["actual_data"])
        inference_data = pickle.loads(decoded_inference_data)
        actual_data = pickle.loads(decoded_actual_data)

        # Construct directory name by removing special characters from model parameters
        base_output_dir = os.path.dirname(filepath)

        file_name = os.path.splitext(row["file_name"])[0]
        model_name = row["model_name"]
        model_params = row["model_params"].translate(str.maketrans('', '', '":, {}'))
        folder_name = f"{file_name}_{model_name}_{model_params}"

        #save_dir = f"{file_name}_{model_name}_{model_params}"
        
        timestamp = f"{int(time.time() * 1000)}"
        #save_dir = os.path.join(save_dir, timestamp)
        save_dir = os.path.join(base_output_dir, folder_name, timestamp)
        print(f"Saving data to directory: {save_dir}")

        # Save the decoded data as CSV files
        to_csv(inference_data, save_dir, "inference_data.csv")
        to_csv(actual_data, save_dir, "actual_data.csv")


# Example usage
# your_result_csv_path = r"/path/to/your_result.csv"
# decode_data(your_result_csv_path)

if __name__ == "__main__":
    import sys
    decode_data(sys.argv[1])