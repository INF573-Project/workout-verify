from pypipeline.schemas.fields import field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List


@dataclass
class DataStageJoints(BaseSchema):
    kpts_detailed: List[dict] = field_persistance()
    hand_kpts_detailed: List[dict] = field_persistance()
    video_output_path: str = field_persistance()
    video_path: str = field_persistance()
    file_name: str = field_persistance()
