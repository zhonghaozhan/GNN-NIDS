import numpy as np
import tensorflow as tf
import time
import psutil

class NIDSInference:
    def __init__(self, model_path):
        """Initialize the NIDS inference engine."""
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("Model loaded successfully")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def preprocess_data(self, data):
        """Preprocess network data for inference."""
        # Add your preprocessing logic here
        return data
    
    def predict(self, data):
        """Run inference on preprocessed data."""
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], data)
            
            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Log performance metrics
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            print(f"Inference time: {inference_time*1000:.2f}ms")
            print(f"Memory usage: {memory_usage:.2f}MB")
            
            return output
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None

def main():
    # Initialize model
    model = NIDSInference("model_nids.tflite")
    
    # Example data (replace with your actual data)
    sample_data = np.random.random((1, 32)).astype(np.float32)
    
    # Preprocess
    processed_data = model.preprocess_data(sample_data)
    
    # Run inference
    result = model.predict(processed_data)
    
    if result is not None:
        print("Prediction:", result)

if __name__ == "__main__":
    main()
