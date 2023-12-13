from pypipeline.schemas.fields import field_persistance, field_perishable
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DataStageHands(BaseSchema):
    keypoints: np.ndarray = field_perishable()
    video_output_path: str = field_persistance()
    video_skeleton_path: str = field_perishable()
    keypoints_hands: np.ndarray = field_perishable()
    hand_kpts_detailed: List[dict] = field_persistance()
    # hand_video_output_path: str = field_perishable()
    video_path: str = field_persistance()
