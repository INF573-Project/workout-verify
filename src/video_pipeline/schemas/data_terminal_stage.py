from pypipeline.schemas.fields import field_persistance
from pypipeline.schemas import BaseSchema
from dataclasses import dataclass


@dataclass
class DataStageTerminal(BaseSchema):
    joint_angles: dict = field_persistance()
    joint_angles_decomposed: dict = field_persistance()
