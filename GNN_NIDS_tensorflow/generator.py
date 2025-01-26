"""
   Copyright 2020 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import csv
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import configparser
from ml_dataset_mapping import attack_type_to_label

# Load normalization parameters
def load_params():
    params = configparser.ConfigParser()
    params._interpolation = configparser.ExtendedInterpolation()
    params.read('./ml_normalization_parameters.ini')
    return params

params = load_params()

# Attack types for ML dataset
attack_names = [
    'Normal',
    'DDoS_UDP',
    'DDoS_ICMP',
    'Ransomware',
    'DDoS_HTTP',
    'SQL_injection',
    'Uploading',
    'DDoS_TCP',
    'Backdoor',
    'Vulnerability_scanner',
    'Port_Scanning',
    'XSS',
    'Password',
    'MITM',
    'Fingerprinting'
]

# MAP THAT TELLS US, GIVEN A FEATURE, ITS POSITION (IDS 2017)
features = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label','Attack_type','Attack_label']
indices = range(len(features))
zip_iterator = zip(features,indices)
features_dict = dict(zip_iterator)

def get_feature(trace, feature_name, parse=True):
    if parse:
        if feature_name == 'Attack_type':
            return trace[feature_name]
        elif feature_name == 'Attack_label':
            return int(trace[feature_name])
        else:
            feature = trace.get(feature_name, 0.0)
            
            if feature is None or feature == '':  # Handle None values
                return 0.0

            if 'ID' in feature_name:
                return feature
            elif 'IP' in feature_name:
                # Return the full IP string for IP features when parse=True
                return str(feature)
            elif feature_name == 'Protocol':
                # Transform protocol to a single numeric value instead of one-hot
                protocol_map = {'tcp': 0.0, 'udp': 1.0, 'icmp': 2.0}
                return protocol_map.get(str(feature).lower(), 0.0)
            else:
                try:
                    value = float(feature)
                    if value != float('+inf') and value != float('nan') and value is not None:
                        return value
                    else:
                        return 0.0
                except:
                    return 0.0
    else:
        # When parse=False, return the raw feature value for IP features
        return str(trace.get(feature_name, '0.0.0.0'))

# constructs a dictionary with all the chosen features of the ids 2017
def get_connection_features(trace, final_feature, type):
    connection_features = {}
    
    for f in chosen_connection_features:
        connection_features[f] = get_feature(trace, f)

    connection_features['Label'] = final_feature
    connection_features['type'] = type
    return connection_features


def traces_to_graph(traces):
    G = nx.MultiDiGraph()

    n = len(traces)
    if n == 0:
        # Create a dummy graph with one IP and one connection
        G.add_node('0.0.0.0', ip=0, type=1)
        connection_features = {feature: 0.0 for feature in chosen_connection_features}
        connection_features['Label'] = np.zeros(len(attack_names))
        G.add_node('con_0', **connection_features)
        G.add_edge('0.0.0.0', 'con_0')
        indices_ip = {'0.0.0.0': 0}
        indices_connection = {'con_0': 0}
        nx.set_node_attributes(G, indices_ip, 'index_ip')
        nx.set_node_attributes(G, indices_connection, 'index_connection')
        G.graph['n_i'] = 1
        G.graph['n_c'] = 1
        return G

    for i in range(n):
        trace = traces[i]

        dst_name = 'Destination IP'
        src_name = 'Source IP'

        dst_ip = get_feature(trace, dst_name, parse=False)
        src_ip = get_feature(trace, src_name, parse=False)

        if dst_ip not in G.nodes():
            G.add_node(dst_ip, ip=transform_ips(dst_ip), type=1)

        if src_ip not in G.nodes():
            G.add_node(src_ip, ip=transform_ips(src_ip), type=1)

        # Get attack type and create one-hot encoding
        attack_type = get_feature(trace, 'Attack_type')
        label_num = attack_names.index(attack_type)
        final_label = np.zeros(len(attack_names))
        final_label[label_num] = 1

        connection_features = get_connection_features(trace, final_label, 2)
        G.add_node('con_' + str(i), **connection_features)

        # these edges connect the ports with the IP node (connecting all the servers together)
        G.add_edge('con_' + str(i), dst_ip)
        G.add_edge(src_ip, 'con_' + str(i))

    # Assign indices and store them in the graph
    indices_ip, indices_connection, n_i, n_c = assign_indices(G)
    nx.set_node_attributes(G, indices_ip, 'index_ip')
    nx.set_node_attributes(G, indices_connection, 'index_connection')
    G.graph['n_i'] = n_i
    G.graph['n_c'] = n_c

    return G


def assign_indices(G):
    indices_ip = {}
    indices_connection = {}
    
    i = 0
    j = 0
    for node in G.nodes():
        if 'con_' in node:
            indices_connection[node] = j
            j += 1
        else:
            indices_ip[node] = i
            i += 1
            
    return indices_ip, indices_connection, i, j

def process_adjacencies(G):
    # Get number of nodes
    n_i = G.graph['n_i']
    n_c = G.graph['n_c']
    
    # Initialize adjacency matrices with max size
    src_ip_to_connection = np.zeros(n_c, dtype=np.int64)
    dst_ip_to_connection = np.zeros(n_c, dtype=np.int64)
    src_connection_to_ip = np.zeros(n_c, dtype=np.int64)
    dst_connection_to_ip = np.zeros(n_c, dtype=np.int64)
    
    # Fill adjacency matrices
    for edge in G.edges():
        src, dst = edge
        if 'con_' in src:
            # Connection to IP
            src_idx = int(G.nodes[src]['index_connection'])
            dst_idx = int(G.nodes[dst]['index_ip'])
            src_connection_to_ip[src_idx] = dst_idx
            dst_connection_to_ip[src_idx] = dst_idx
        else:
            # IP to Connection
            src_idx = int(G.nodes[src]['index_ip'])
            dst_idx = int(G.nodes[dst]['index_connection'])
            src_ip_to_connection[dst_idx] = src_idx
            dst_ip_to_connection[dst_idx] = src_idx
            
    return src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip

def graph_to_dict(G):
    # Process adjacency matrices
    src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip = process_adjacencies(G)
    
    # Get connection features
    connection_nodes = sorted([n for n in G.nodes() if 'con_' in n], 
                            key=lambda x: int(x.split('_')[1]))  # Sort by connection index
    n_connections = len(connection_nodes)
    
    # Initialize feature matrix
    feature_connection = np.zeros((n_connections, len(chosen_connection_features)))
    labels = np.zeros((n_connections, len(attack_names)))
    
    # Fill feature matrix
    for i, node in enumerate(connection_nodes):
        node_data = G.nodes[node]
        for j, feature in enumerate(chosen_connection_features):
            feature_connection[i, j] = node_data[feature]
        labels[i] = node_data['Label']
    
    # Pad arrays to max_n_connections if needed
    max_n_connections = 200  # This should match the window size in the generator
    if n_connections < max_n_connections:
        pad_size = max_n_connections - n_connections
        feature_connection = np.pad(feature_connection, ((0, pad_size), (0, 0)))
        labels = np.pad(labels, ((0, pad_size), (0, 0)))
        src_ip_to_connection = np.pad(src_ip_to_connection, (0, pad_size))
        dst_ip_to_connection = np.pad(dst_ip_to_connection, (0, pad_size))
        src_connection_to_ip = np.pad(src_connection_to_ip, (0, pad_size))
        dst_connection_to_ip = np.pad(dst_connection_to_ip, (0, pad_size))
    
    return {
        'feature_connection': feature_connection.astype('float32'),
        'src_ip_to_connection': src_ip_to_connection,
        'dst_ip_to_connection': dst_ip_to_connection,
        'src_connection_to_ip': src_connection_to_ip,
        'dst_connection_to_ip': dst_connection_to_ip,
        'n_i': G.graph['n_i'],  # Return as Python int
        'n_c': G.graph['n_c'],  # Return as Python int
        'label': labels.astype('float32')
    }

def load_data(path):
    """Load data from CSV files and convert to tensors."""
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    elif isinstance(path, np.ndarray):
        path = path.item().decode('utf-8')
    
    files = glob.glob(path + '/*.csv')
    print(f"Found {len(files)} CSV files: {files}")
    
    all_features = []
    all_labels = []
    
    for file in files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file)
        print(f"Loaded DataFrame with {len(df)} rows")
        
        # Process data in windows of 200 rows
        window_size = 200
        for i in range(0, len(df), window_size):
            window = df.iloc[i:i+window_size]
            if len(window) == window_size:  # Only process complete windows
                try:
                    print(f"Processing window {i//window_size + 1} with {len(window)} rows")
                    # Convert window rows to list of dictionaries
                    traces = window.to_dict('records')
                    
                    # Create graph from traces
                    G = traces_to_graph(traces)
                    features = graph_to_dict(G)
                    
                    # Extract features and labels
                    x = {
                        'feature_connection': features['feature_connection'].astype(np.float32),
                        'n_i': np.array(features['n_i'], dtype=np.int64),
                        'n_c': np.array(features['n_c'], dtype=np.int64),
                        'src_ip_to_connection': features['src_ip_to_connection'].astype(np.int64),
                        'dst_ip_to_connection': features['dst_ip_to_connection'].astype(np.int64),
                        'src_connection_to_ip': features['src_connection_to_ip'].astype(np.int64),
                        'dst_connection_to_ip': features['dst_connection_to_ip'].astype(np.int64)
                    }
                    y = features['label'].astype(np.float32)
                    
                    # Print shapes for debugging
                    print("Feature shapes:")
                    print(f"feature_connection: {x['feature_connection'].shape}")
                    print(f"n_i: {x['n_i'].shape}")
                    print(f"n_c: {x['n_c'].shape}")
                    print(f"src_ip_to_connection: {x['src_ip_to_connection'].shape}")
                    print(f"dst_ip_to_connection: {x['dst_ip_to_connection'].shape}")
                    print(f"src_connection_to_ip: {x['src_connection_to_ip'].shape}")
                    print(f"dst_connection_to_ip: {x['dst_connection_to_ip'].shape}")
                    print(f"labels: {y.shape}")
                    
                    all_features.append(x)
                    all_labels.append(y)
                    
                    print(f"Successfully processed window {i//window_size + 1}")
                except Exception as e:
                    print(f"Error processing window at index {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Find maximum dimensions
    max_n_i = max(f['n_i'] for f in all_features)
    max_n_c = max(f['n_c'] for f in all_features)
    
    # Pad each feature to max dimensions
    padded_features = []
    for x in all_features:
        # Calculate padding sizes
        n_c_pad = max(0, max_n_c - x['feature_connection'].shape[0])
        n_i_pad = max(0, max_n_i - len(x['src_connection_to_ip']))
        
        padded_x = {
            'feature_connection': np.pad(x['feature_connection'], 
                                      ((0, n_c_pad), (0, 0))),
            'n_i': x['n_i'],
            'n_c': x['n_c'],
            'src_ip_to_connection': np.pad(x['src_ip_to_connection'], 
                                         (0, n_c_pad)),
            'dst_ip_to_connection': np.pad(x['dst_ip_to_connection'], 
                                         (0, n_c_pad)),
            'src_connection_to_ip': np.pad(x['src_connection_to_ip'], 
                                         (0, n_i_pad)),
            'dst_connection_to_ip': np.pad(x['dst_connection_to_ip'], 
                                         (0, n_i_pad))
        }
        padded_features.append(padded_x)
    
    # Stack all features and labels
    stacked_features = {
        'feature_connection': tf.convert_to_tensor([f['feature_connection'] for f in padded_features]),
        'n_i': tf.convert_to_tensor([f['n_i'] for f in padded_features]),
        'n_c': tf.convert_to_tensor([f['n_c'] for f in padded_features]),
        'src_ip_to_connection': tf.convert_to_tensor([f['src_ip_to_connection'] for f in padded_features]),
        'dst_ip_to_connection': tf.convert_to_tensor([f['dst_ip_to_connection'] for f in padded_features]),
        'src_connection_to_ip': tf.convert_to_tensor([f['src_connection_to_ip'] for f in padded_features]),
        'dst_connection_to_ip': tf.convert_to_tensor([f['dst_connection_to_ip'] for f in padded_features])
    }
    stacked_labels = tf.convert_to_tensor(all_labels)
    
    # Print final stacked shapes
    print("\nFinal stacked shapes:")
    for key, tensor in stacked_features.items():
        print(f"{key}: {tensor.shape}")
    print(f"labels: {stacked_labels.shape}")
    
    return stacked_features, stacked_labels

chosen_connection_features = ['Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
                   'Total Length of Fwd Packets', 'Fwd Packet Length Mean',
                   'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
                   'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
                   'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
                   'PSH Flag Count', 'URG Flag Count']

def normalization_function(feature, labels):
    """
    Normalize features using mean and standard deviation from parameters.
    """
    # Reload parameters
    global params
    params = load_params()
    
    # Get feature connection tensor
    feature_connection = feature['feature_connection']
    
    # Create tensors for means and stds
    means = tf.constant([float(params['MEANS'][feature_name]) for feature_name in chosen_connection_features], 
                       dtype=tf.float32)
    stds = tf.constant([float(params['STDS'][feature_name]) for feature_name in chosen_connection_features], 
                      dtype=tf.float32)
    
    # Reshape means and stds to match feature dimensions
    means = tf.reshape(means, [1, len(chosen_connection_features)])
    stds = tf.reshape(stds, [1, len(chosen_connection_features)])
    
    # Normalize features using broadcasting
    normalized_feature_connection = (feature_connection - means) / (stds + 1e-7)
    
    # Update feature dictionary with normalized features
    normalized_features = {
        'feature_connection': normalized_feature_connection,
        'n_i': feature['n_i'],
        'n_c': feature['n_c'],
        'src_ip_to_connection': feature['src_ip_to_connection'],
        'dst_ip_to_connection': feature['dst_ip_to_connection'],
        'src_connection_to_ip': feature['src_connection_to_ip'],
        'dst_connection_to_ip': feature['dst_connection_to_ip']
    }
    
    return normalized_features, labels

def transform_ips(ip):
    """Transform IP address to integer."""
    try:
        # Split IP into octets
        octets = ip.split('.')
        # Convert each octet to integer and combine
        result = (int(octets[0]) << 24) + (int(octets[1]) << 16) + (int(octets[2]) << 8) + int(octets[3])
        return result
    except:
        return 0  # Return 0 for invalid IPs

def input_fn(data_path, validation=False):
    """Creates an input pipeline using tf.data.Dataset."""
    
    # Load all data into memory
    features, labels = load_data(data_path)
    
    # Create dataset from tensors
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    
    # Apply normalization function
    ds = ds.map(normalization_function)
    
    # Shuffle and repeat for training
    if not validation:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.repeat()
    
    # Batch the dataset
    if validation:
        ds = ds.batch(1)
    else:
        ds = ds.batch(16)
    
    return ds
