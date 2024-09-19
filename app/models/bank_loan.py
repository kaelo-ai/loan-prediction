import pickle
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(current_dir, 'artifacts', 'best_model.pkl')
proprocessor_path = os.path.join(current_dir, 'artifacts', 'proprocessor.pkl')

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)



# Export the model 
__all__ = ['model']