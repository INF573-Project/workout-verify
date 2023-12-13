from pypipeline.schemas.fields import field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List


@dataclass
class DataStageClassify(BaseSchema):
    kpts_detailed: List[dict] = field_persistance()
    hand_kpts_detailed: List[dict] = field_persistance()
    workouts: List[dict] = field_persistance()
    video_output_path: str = field_persistance()
    video_path: str = field_persistance()
