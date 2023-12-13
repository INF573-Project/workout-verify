from pypipeline.schemas.fields import field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
import numpy as np


@dataclass
class DataStageInference(BaseSchema):
    keypoints: np.ndarray = field_persistance()
    video_output_path: str = field_persistance()
    video_skeleton_path: str = field_persistance()
    keypoints_hands: np.ndarray = field_persistance()
    # hand_video_output_path: str = field_persistance()
    video_path: str = field_persistance()
