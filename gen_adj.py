import pandas as pd
import numpy as np
import pickle

def generate_adjacency_matrix(input_csv_path, output_pkl_path, threshold=0.1):
    """
    Generate adjacency matrix based on correlation of variables in the dataset.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_pkl_path (str): Path to save the adjacency matrix in .pkl format.
        threshold (float): Correlation threshold to create connections in adjacency matrix.

    Returns:
        None
    """
    # Load dataset
    data = pd.read_csv(input_csv_path)
    
    # Drop non-numeric columns (e.g., 'date') to compute correlation
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    numeric_data = data[numeric_columns]
    
    if numeric_data.empty:
        raise ValueError("The dataset does not contain any numeric columns for correlation computation.")
    
    # Compute correlation matrix
    correlation_matrix = numeric_data.corr().values
    
    # Generate adjacency matrix based on threshold
    adjacency_matrix = (np.abs(correlation_matrix) > threshold).astype(float)
    
    # Remove self-loops (diagonal values)
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Create the required dictionary structure
    adj_data = {
        "sensor_ids": list(numeric_columns),
        "sensor_id_to_ind": {col: i for i, col in enumerate(numeric_columns)},
        "adj": adjacency_matrix
    }
    
    # Save to .pkl file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(adj_data, f)
    
    print(f"Adjacency matrix saved to {output_pkl_path}")
    print(f"Number of nodes: {len(numeric_columns)}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    print(f"Threshold used: {threshold}")
    print(f"Sensor IDs: {adj_data['sensor_ids']}")
    print(f"Sensor ID to Index Mapping: {adj_data['sensor_id_to_ind']}")
    print(f"Adjacency Matrix:\n{adj_data['adj']}")

# Paths for input and output
input_csv_path = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0.csv'
output_pkl_path = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0_adj_mx.pkl'

# Generate and save adjacency matrix
generate_adjacency_matrix(input_csv_path, output_pkl_path, threshold=0.1)



######################
import pickle

with open('/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0_adj_mx.pkl', 'rb') as f:
    adj_data = pickle.load(f)

print("Sensor IDs:", adj_data["sensor_ids"])
print("Sensor ID to Index Mapping:", adj_data["sensor_id_to_ind"])
print("Adjacency Matrix:\n", adj_data["adj"])





import pickle

# 检查 .pkl 文件内容
with open('/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0_adj_mx.pkl', 'rb') as f:
    data = pickle.load(f)

print("Content of .pkl file:", data)

