import tensorflow as tf
from keras.models import load_model

def export(input_h5_file, export_path):
    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = load_model(input_h5_file)
    model.save(export_path, save_format='tf')
    print(f"SavedModel created at {export_path}")

if __name__ == "__main__":

    input_h5_file = 'snapshots/fcn_model.h5'
    export_path = 'flower_classifier/1'
    export(input_h5_file, export_path)