from pypipeline.schemas.fields import field_persistance, field_perishable
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List


@dataclass
class DataStageTerminal(BaseSchema):
    kpts_detailed: List[dict] = field_persistance()
    hand_kpts_detailed: List[dict] = field_persistance()
    workouts: List[dict] = field_perishable()
    video_output_path: str = field_perishable()
    video_path: str = field_perishable()
    joints_history: dict = field_persistance()
    workout_advice: List[dict] = field_persistance()
    file_name: str = field_persistance()
    workout_rep_extrema: List[dict] = field_persistance()
