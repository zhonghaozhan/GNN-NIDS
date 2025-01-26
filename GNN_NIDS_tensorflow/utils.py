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
import tensorflow_addons as tfa
from GNN import GNN
import os

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

def _get_compiled_model(params):
    model = GNN(params)
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(params['HYPERPARAMETERS']['learning_rate']),
                                                                int(params['HYPERPARAMETERS']['decay_steps']),
                                                                float(params['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    
    # Basic metrics
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
        tf.keras.metrics.AUC(curve='ROC', name='roc_auc')
    ]
    
    # Add per-class metrics for all 15 classes
    for i in range(15):
        metrics.extend([
            tf.keras.metrics.Recall(class_id=i, name=f'recall_{i}'),
            tf.keras.metrics.Precision(class_id=i, name=f'precision_{i}')
        ])

    model.compile(optimizer=optimizer,
                 loss=loss_object,
                 metrics=metrics,
                 run_eagerly=True)  # Enable eager execution for debugging

    return model


import glob
def make_or_restore_model(params):
    # Either restore the latest model, or create a fresh one
    checkpoint_dir = os.path.abspath(params['DIRECTORIES']['logs'] + '/ckpt')
    # if there is no checkpoint available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print("Creating a new model")
    return _get_compiled_model(params)
