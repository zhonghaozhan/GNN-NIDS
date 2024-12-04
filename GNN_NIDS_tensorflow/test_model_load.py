import tensorflow as tf
import tensorflow_addons as tfa
import os

class CustomF1Score(tfa.metrics.F1Score):
    def __init__(self, name='macro_F1', **kwargs):
        super().__init__(name=name, **kwargs)

def test_model_load():
    # Path to the saved model
    model_path = os.path.join('logs', 'ckpt', 'model_400')
    
    try:
        # Define custom objects
        custom_objects = {
            'Addons>F1Score': CustomF1Score,
            'macro_F1': lambda: CustomF1Score(num_classes=15, average='macro')
        }
        
        # Load the model with custom objects
        print("Loading model from:", model_path)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        # Print model summary
        print("\nModel loaded successfully!")
        print("\nModel Summary:")
        model.summary()
        
        return True
    except Exception as e:
        print("Error loading model:", str(e))
        return False

if __name__ == "__main__":
    test_model_load()
