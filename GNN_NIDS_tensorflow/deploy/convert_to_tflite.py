import tensorflow as tf
import os

def convert_to_tflite(saved_model_dir, output_file):
    """Convert TensorFlow model to TFLite format."""
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantize the model to reduce size and improve CPU performance
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to: {output_file}")
    print(f"Model size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # Path to your saved model
    saved_model_dir = "../logs/ckpt/model_400"
    
    # Output TFLite model path
    output_file = "model_nids.tflite"
    
    convert_to_tflite(saved_model_dir, output_file)
