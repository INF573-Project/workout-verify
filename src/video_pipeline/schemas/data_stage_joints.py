from pypipeline.schemas.fields import field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List


@dataclass
class DataStageJoints(BaseSchema):
    joints_kpts_detailed: List[dict] = field_persistance()
    joints_video_output_path: str = field_persistance()
