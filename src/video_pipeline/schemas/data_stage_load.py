from pypipeline.schemas.fields import field_perishable, field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass


@dataclass
class DataStageLoad(BaseSchema):
    video_path: str = field_persistance()
    hand_video_output_path: str = field_perishable()
    joints_video_output_path: str = field_persistance()
    video_skeleton_path: str = field_persistance()

@dataclass
class DataStageOutputLoad(BaseSchema):
    video_path: str = field_perishable()
    hand_video_output_path: str = field_perishable()
    joints_video_output_path: str = field_persistance()
    video_skeleton_path: str = field_persistance()