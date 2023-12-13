import pickle

# Local imports
from src.utils.joint_features import JointFeatures

class WorkoutClassifier:

  def __init__(self, model_path: str):
    with open(model_path, 'rb') as f:
      self.clf = pickle.load(f)
    
    self.joint_features = JointFeatures()
  

  def predict(self, joint_history: dict):
    dataset = self.joint_features.get_dataset(joint_history)
    
    return self.clf.predict(dataset)