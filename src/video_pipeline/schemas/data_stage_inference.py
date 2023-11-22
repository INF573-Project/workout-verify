from pypipeline.schemas.fields import field_persistance, field_perishable
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
import numpy as np


@dataclass
class DataStageInference(BaseSchema):
    keypoints: np.ndarray = field_perishable()
    video_output_path: str = field_persistance()
    video_skeleton_path: str = field_perishable()
