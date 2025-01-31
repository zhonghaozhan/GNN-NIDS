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

import configparser
import time
import numpy as np
import os
import tensorflow as tf
from utils import make_or_restore_model
from generator import input_fn
import configparser

# Enable MPS (Metal Performance Shaders) for Apple Silicon
try:
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
    else:
        print("No GPU found. Checking for Apple Metal...")
        if tf.config.list_physical_devices('MPS'):
            print("Apple Metal device found. Using MPS backend.")
            tf.config.set_visible_devices(tf.config.list_physical_devices('MPS')[0], 'MPS')
        else:
            print("No accelerator found. Using CPU.")
except:
    print("Error checking devices. Falling back to CPU.")

# Enable memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU device found:", physical_devices[0].name)
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Performance optimizations
tf.config.optimizer.set_jit(True)  # Enable XLA optimization
tf.data.experimental.enable_debug_mode()

# Set up mixed precision policy for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

model = make_or_restore_model(params=params)

# callbacks to save the model
path_logs = os.path.abspath(params['DIRECTORIES']['logs'])
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=path_logs + "/ckpt/model_{epoch:02d}",
        save_weights_only=False,
        save_freq='epoch',
        monitor='loss',
        save_best_only=False,
        save_format='tf'
    ),
    tf.keras.callbacks.TensorBoard(log_dir=path_logs + "/logs", update_freq=1000)
]

train_dataset = input_fn(data_path=os.path.abspath(params["DIRECTORIES"]["train"]), validation=False)
val_dataset = input_fn(data_path=os.path.abspath(params["DIRECTORIES"]["validation"]), validation=True)

# Check if datasets are empty
print("Checking datasets...")
try:
    first_batch = next(iter(train_dataset))
    print("First training batch shapes:")
    print("Features:", {k: v.shape for k, v in first_batch[0].items()})
    print("Labels:", first_batch[1].shape)
except StopIteration:
    print("Error: Training dataset is empty!")
    raise

try:
    first_batch = next(iter(val_dataset))
    print("First validation batch shapes:")
    print("Features:", {k: v.shape for k, v in first_batch[0].items()})
    print("Labels:", first_batch[1].shape)
except StopIteration:
    print("Error: Validation dataset is empty!")
    raise

# Training the model
model.fit(train_dataset,
          validation_data=val_dataset,
          validation_steps=600,
          steps_per_epoch=1600,
          batch_size=16,
          epochs=int(params['HYPERPARAMETERS']['epochs']),
          callbacks=callbacks,
          use_multiprocessing=True)
