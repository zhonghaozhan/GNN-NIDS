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

import tensorflow as tf
from generator import chosen_connection_features

class GNN(tf.keras.Model):

    def __init__(self, config):
        super(GNN, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # Feature transformation layers
        self.feature_transform = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']), activation=tf.nn.relu)
        ])

        # Message passing neural networks
        self.message_func1 = tf.keras.Sequential([
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']), activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']))
        ])
        
        self.message_func2 = tf.keras.Sequential([
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']), activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']))
        ])

        # RNN cells for update functions
        self.ip_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['node_state_dim']))
        self.connection_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['node_state_dim']))

        # Readout layer (final classification)
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.keras.regularizers.l2(float(self.config['HYPERPARAMETERS']['l2']))),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['num_classes']), activation=tf.nn.softmax)
        ])

    def message_passing_step(self, ip_state, connection_state, src_ip_to_connection, dst_ip_to_connection, 
                           src_connection_to_ip, dst_connection_to_ip, n_ips, n_connections):
        """Performs one step of message passing."""
        
        # IP TO CONNECTION
        # Gather node states
        ip_node_gather = tf.gather(ip_state, src_ip_to_connection)
        connection_gather = tf.gather(connection_state, dst_ip_to_connection)

        # Transform and concat features
        nn_input = tf.concat([ip_node_gather, connection_gather], axis=-1)

        # Message passing
        ip_message = self.message_func1(nn_input)
        ip_mean = tf.math.unsorted_segment_mean(ip_message, dst_ip_to_connection, n_connections)

        # CONNECTION TO IP
        connection_node_gather = tf.gather(connection_state, src_connection_to_ip)
        ip_gather = tf.gather(ip_state, dst_connection_to_ip)

        # Transform and concat features
        nn_input = tf.concat([connection_node_gather, ip_gather], axis=-1)

        # Message passing
        connection_messages = self.message_func2(nn_input)
        connection_mean = tf.math.unsorted_segment_mean(connection_messages, dst_connection_to_ip, n_ips)

        return ip_mean, connection_mean

    @tf.function
    def call(self, inputs):
        # Get inputs
        feature_connection = tf.cast(inputs['feature_connection'], tf.float32)
        n_ips = tf.cast(inputs['n_i'], tf.int32)
        n_connections = tf.cast(inputs['n_c'], tf.int32)
        src_ip_to_connection = tf.cast(inputs['src_ip_to_connection'], tf.int32)
        dst_ip_to_connection = tf.cast(inputs['dst_ip_to_connection'], tf.int32)
        src_connection_to_ip = tf.cast(inputs['src_connection_to_ip'], tf.int32)
        dst_connection_to_ip = tf.cast(inputs['dst_connection_to_ip'], tf.int32)

        # Get state dimension
        state_dim = int(self.config['HYPERPARAMETERS']['node_state_dim'])

        # Initialize states
        batch_size = tf.shape(feature_connection)[0]
        max_n_ips = tf.cast(tf.reduce_max(n_ips), tf.int32)
        max_n_connections = tf.cast(tf.reduce_max(n_connections), tf.int32)
        
        # Initialize states with zeros
        ip_state = tf.zeros([batch_size, max_n_ips, state_dim])
        connection_state = self.feature_transform(feature_connection)

        # Message passing iterations
        for _ in range(int(self.config['HYPERPARAMETERS']['T'])):
            # Process each batch element
            for b in tf.range(batch_size):
                # Get the number of IPs and connections for this batch
                n_i = tf.maximum(n_ips[b], 1)  # Ensure at least one node
                n_c = tf.maximum(n_connections[b], 1)  # Ensure at least one connection
                
                # Get batch slice
                batch_ip_state = ip_state[b, :n_i]
                batch_connection_state = connection_state[b, :n_c]
                batch_src_ip_to_connection = src_ip_to_connection[b, :n_c]
                batch_dst_ip_to_connection = dst_ip_to_connection[b, :n_c]
                batch_src_connection_to_ip = src_connection_to_ip[b, :n_i]
                batch_dst_connection_to_ip = dst_connection_to_ip[b, :n_i]

                # Perform message passing
                ip_mean, connection_mean = self.message_passing_step(
                    batch_ip_state, batch_connection_state,
                    batch_src_ip_to_connection, batch_dst_ip_to_connection,
                    batch_src_connection_to_ip, batch_dst_connection_to_ip,
                    n_i, n_c
                )

                # Update states
                # Create initial states with correct shapes
                ip_initial_state = tf.zeros([n_i, state_dim])
                connection_initial_state = tf.zeros([n_c, state_dim])

                # Update IP states
                ip_state_new, _ = self.ip_update(connection_mean, [ip_initial_state])
                
                # Update connection states
                connection_state_new, _ = self.connection_update(ip_mean, [connection_initial_state])

                # Pad back to max size
                ip_state_new = tf.pad(ip_state_new, [[0, max_n_ips - n_i], [0, 0]])
                connection_state_new = tf.pad(connection_state_new, [[0, max_n_connections - n_c], [0, 0]])

                # Update the states for this batch
                ip_state = tf.tensor_scatter_nd_update(
                    ip_state,
                    tf.stack([tf.repeat(b, max_n_ips), tf.range(max_n_ips)], axis=1),
                    ip_state_new
                )
                connection_state = tf.tensor_scatter_nd_update(
                    connection_state,
                    tf.stack([tf.repeat(b, max_n_connections), tf.range(max_n_connections)], axis=1),
                    connection_state_new
                )

        # Readout
        nn_output = self.readout(connection_state)
        return nn_output
