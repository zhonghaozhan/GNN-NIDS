import pandas as pd
import numpy as np
from generator import traces_to_graph, graph_to_dict, get_connection_features, features
import tensorflow as tf
from GNN import GNN
import configparser
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from ml_dataset_mapping import map_attack_type, map_label_to_attack, attack_type_to_label

# Create features dictionary for indexing
features_dict = {feature: i for i, feature in enumerate(features)}

def convert_to_ids2017_format(row):
    """Convert a row from ML-dataset to IDS2017 format."""
    trace = [0.0] * len(features)  # Initialize with zeros
    
    # Map the fields we have
    trace[features_dict['Flow ID']] = f"{row['ip.src_host']}_{row['ip.dst_host']}"
    trace[features_dict['Source IP']] = str(row['ip.src_host'])
    trace[features_dict['Destination IP']] = str(row['ip.dst_host']) if str(row['ip.dst_host']) != '0.0' else '0.0.0.0'
    trace[features_dict['Source Port']] = float(row['tcp.srcport']) if not pd.isna(row['tcp.srcport']) else 0.0
    trace[features_dict['Destination Port']] = float(row['tcp.dstport']) if not pd.isna(row['tcp.dstport']) else 0.0
    trace[features_dict['Protocol']] = 6.0  # Default to TCP
    
    # Convert timestamp to epoch time
    try:
        timestamp_str = str(row['frame.time']).strip()
        timestamp = pd.to_datetime(timestamp_str).timestamp()
        trace[features_dict['Timestamp']] = float(timestamp)
    except (ValueError, AttributeError):
        trace[features_dict['Timestamp']] = 0.0
    
    # Set numeric fields
    numeric_fields = {
        'Flow Duration': 'udp.time_delta',  # Using UDP time delta as proxy for flow duration
        'Total Fwd Packets': 'tcp.len',  # Using tcp.len as a proxy for packet count
        'Total Backward Packets': 'tcp.len',
        'Total Length of Fwd Packets': 'tcp.len',
        'Total Length of Bwd Packets': 'tcp.len',
        'Fwd Packet Length Mean': 'tcp.len',
        'Fwd Packet Length Std': 0.0,
        'Bwd Packet Length Min': 'tcp.len',
        'Bwd Packet Length Std': 0.0,
        'Flow IAT Mean': 'udp.time_delta',
        'Flow IAT Std': 0.0,
        'Flow IAT Min': 'udp.time_delta',
        'Fwd IAT Min': 'udp.time_delta',
        'Bwd IAT Mean': 'udp.time_delta',
        'Fwd PSH Flags': 'tcp.flags',  # Will extract PSH flag from tcp.flags
        'SYN Flag Count': 'tcp.connection.syn',
        'PSH Flag Count': 'tcp.flags',
        'ACK Flag Count': 'tcp.flags.ack',
        'Subflow Fwd Packets': 'tcp.len',
        'Subflow Fwd Bytes': 'tcp.len',
        'Subflow Bwd Bytes': 'tcp.len',
        'Init_Win_bytes_forward': 'tcp.len',  # Using tcp.len as proxy
        'Flow Packets/s': 'tcp.len',  # Will calculate based on tcp.len and time delta
        'Average Packet Size': 'tcp.len',
        'Active Min': 0.0,
        'Active Mean': 0.0
    }
    
    for field, col in numeric_fields.items():
        if field in features_dict:
            if isinstance(col, str):
                if col == 'tcp.flags':
                    # Extract PSH flag from tcp.flags (if PSH flag is set, bit 3 will be 1)
                    flags = int(row[col]) if not pd.isna(row[col]) else 0
                    psh_flag = 1 if flags & 0x08 else 0
                    trace[features_dict[field]] = float(psh_flag)
                elif col == 'udp.time_delta':
                    # Convert time delta to microseconds
                    try:
                        delta = float(row[col]) if not pd.isna(row[col]) else 0.0
                        trace[features_dict[field]] = delta * 1000000  # Convert to microseconds
                    except (ValueError, TypeError):
                        trace[features_dict[field]] = 0.0
                elif col == 'tcp.len':
                    # Handle tcp.len field
                    try:
                        length = float(row[col]) if not pd.isna(row[col]) else 0.0
                        if field == 'Flow Packets/s' and not pd.isna(row['udp.time_delta']):
                            time_delta = float(row['udp.time_delta'])
                            if time_delta > 0:
                                trace[features_dict[field]] = length / time_delta
                            else:
                                trace[features_dict[field]] = 0.0
                        else:
                            trace[features_dict[field]] = length
                    except (ValueError, TypeError):
                        trace[features_dict[field]] = 0.0
                else:
                    try:
                        trace[features_dict[field]] = float(row[col]) if not pd.isna(row[col]) else 0.0
                    except (ValueError, TypeError):
                        trace[features_dict[field]] = 0.0
            else:
                trace[features_dict[field]] = float(col)
    
    # Map attack type to label index
    attack_type = row['Attack_type'] if row['Attack_label'] == 1 else 'BENIGN'
    label_idx = map_attack_type(attack_type)
    trace.append(label_idx)
    
    return trace

def load_and_preprocess_data(csv_path):
    """Load and preprocess the ML-dataset.csv into the format expected by the model."""
    # Read the CSV file
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert DataFrame rows to IDS2017 format
    traces = []
    for _, row in df.iterrows():
        # Skip rows with invalid IP addresses
        if pd.isna(row['ip.src_host']) or str(row['ip.src_host']) == '0' or str(row['ip.src_host']) == '0.0':
            continue
            
        try:
            trace = convert_to_ids2017_format(row)
            traces.append(trace)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return traces

def process_batch(model, batch_traces):
    """Process a batch of traces and return predictions."""
    if not batch_traces:
        return None, None
    
    # Convert traces to graph format
    try:
        graph = traces_to_graph(batch_traces)
        features_dict = graph_to_dict(graph)
        
        # Extract label from features_dict
        labels = features_dict.pop('label')
        
        # n_i and n_c should already be Python integers from graph_to_dict
        n_nodes = features_dict['n_i']
        n_connections = features_dict['n_c']
        
        # Ensure dimensions match for adjacency matrices
        src_ip_to_connection = features_dict['src_ip_to_connection'][:n_nodes, :n_connections]
        dst_ip_to_connection = features_dict['dst_ip_to_connection'][:n_nodes, :n_connections]
        src_connection_to_ip = features_dict['src_connection_to_ip'][:n_connections, :n_nodes]
        dst_connection_to_ip = features_dict['dst_connection_to_ip'][:n_connections, :n_nodes]
        
        # Convert inputs to tensors with correct dtypes and shapes
        feature_connection = tf.convert_to_tensor(features_dict['feature_connection'][:n_connections], dtype=tf.float32)
        feature_connection = tf.reshape(feature_connection, [-1, 26])  # Flatten to 2D
        
        # Convert adjacency matrices to tensors and flatten them
        src_ip_to_connection = tf.convert_to_tensor(src_ip_to_connection, dtype=tf.int64)
        src_ip_to_connection = tf.reshape(src_ip_to_connection, [-1])
        
        dst_ip_to_connection = tf.convert_to_tensor(dst_ip_to_connection, dtype=tf.int64)
        dst_ip_to_connection = tf.reshape(dst_ip_to_connection, [-1])
        
        src_connection_to_ip = tf.convert_to_tensor(src_connection_to_ip, dtype=tf.int64)
        src_connection_to_ip = tf.reshape(src_connection_to_ip, [-1])
        
        dst_connection_to_ip = tf.convert_to_tensor(dst_connection_to_ip, dtype=tf.int64)
        dst_connection_to_ip = tf.reshape(dst_connection_to_ip, [-1])
        
        inputs = {
            'feature_connection': feature_connection,
            'n_i': tf.convert_to_tensor(n_nodes, dtype=tf.int64),
            'n_c': tf.convert_to_tensor(n_connections, dtype=tf.int64),
            'src_ip_to_connection': src_ip_to_connection,
            'dst_ip_to_connection': dst_ip_to_connection,
            'src_connection_to_ip': src_connection_to_ip,
            'dst_connection_to_ip': dst_connection_to_ip
        }
        
        # Get predictions using the loaded model
        predictions = model(inputs)
        
        return predictions.numpy(), labels
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None, None

def evaluate_model(model, traces, batch_size=32):
    """Evaluate model on traces."""
    all_predictions = []
    all_labels = []
    
    # Process traces in batches
    num_batches = len(traces) // batch_size + (1 if len(traces) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(traces))
        batch_traces = traces[start_idx:end_idx]
        
        # Get predictions for batch
        predictions, labels = process_batch(model, batch_traces)
        
        # Get true labels
        true_labels = [map_attack_type(trace['Attack_type']) for trace in batch_traces]
        
        all_predictions.extend(predictions)
        all_labels.extend(true_labels)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Generate classification report with proper label names
    target_names = [map_label_to_attack(i) for i in range(len(attack_type_to_label))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main(model_path, data_path, config_path='./config.ini'):
    """Evaluate the model on the given dataset."""
    # Load configuration
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(config_path)
    
    # Load saved model
    try:
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    traces = load_and_preprocess_data(data_path)
    print(f"Loaded {len(traces)} valid traces")
    
    evaluate_model(model, traces)

if __name__ == "__main__":
    main(
        model_path="GNN_NIDS_tensorflow/model_backups/best_model_backup_20241204",  # Use backup model
        data_path="GNN_NIDS_tensorflow/preprocessed_IDS2017/ML-dataset.csv",  # Evaluation data
        config_path='GNN_NIDS_tensorflow/config.ini'  # Config file
    )
