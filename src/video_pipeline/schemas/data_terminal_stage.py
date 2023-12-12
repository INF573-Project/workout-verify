from pypipeline.schemas.fields import field_persistance, field_perishable
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List


@dataclass
class DataStageTerminal(BaseSchema):
    hand_kpts_detailed: List[dict] = field_persistance()
    hand_video_output_path: str = field_perishable()
    joints_kpts_detailed: List[dict] = field_persistance()
    joints_video_output_path: str = field_perishable()
