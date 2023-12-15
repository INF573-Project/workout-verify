from pypipeline.schemas.fields import field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass
from typing import List


@dataclass
class DataStageAdvice(BaseSchema):
    file_name: str = field_persistance()
    kpts_detailed: List[dict] = field_persistance()
    hand_kpts_detailed: List[dict] = field_persistance()
    workouts: List[dict] = field_persistance()
    video_output_path: str = field_persistance()
    video_path: str = field_persistance()
    joints_history: dict = field_persistance()
    workout_advice: List[dict] = field_persistance()
    workout_rep_extrema: List[dict] = field_persistance()