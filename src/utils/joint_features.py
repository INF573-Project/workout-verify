import numpy as np
from collections import defaultdict


class JointFeatures:

  def __init__(self):
    self.window_size = 64
    self.discriminative_joints = set(
      ["knee_right", "knee_left", "elbow_left", "elbow_right"])
    self.new_discriminative = set(["knee", "elbow"])
  

  def _compute_deltas(self, joint_history: dict) -> dict:
    joint_deltas = defaultdict(list)

    for joint in joint_history:
      for i, angle in enumerate(joint_history[joint]):
        if i == 0: continue

        joint_deltas[joint].append(angle - joint_history[joint][i-1])
    
    return joint_deltas


  def _extract_features(self, joint_history: dict) -> dict:
    joint_deltas = self._compute_deltas({
      key: joint_history[key] for key in self.discriminative_joints})
    
    for joint in self.new_discriminative:
      joint_deltas[joint] = (np.array(joint_history[f"{joint}_right"]) + np.array(joint_history[f"{joint}_left"])) / 2

    del joint_deltas["knee_left"]
    del joint_deltas["knee_right"]
    del joint_deltas["elbow_right"]
    del joint_deltas["elbow_left"]
    
    joint_features = defaultdict(list)

    for joint in joint_deltas:
      i = 0
      j = i + self.window_size

      while j < len(joint_deltas[joint]):
        joint_features[joint].append(np.array(joint_deltas[joint])[i:j])
        i += 1
        j += 1

    return joint_features


  def _build_dataset(self, joint_features: dict):
    dataset = []

    for i in range(len(joint_features[list(self.new_discriminative)[0]])):
        combined_features = []
        
        for joint in self.new_discriminative:
            combined_features.extend(joint_features[joint][i])

        dataset.append(combined_features)
    
    return dataset


  def get_dataset(self, joint_history: dict):
    joint_features = self._extract_features(joint_history)
    dataset = self._build_dataset(joint_features)

    return dataset