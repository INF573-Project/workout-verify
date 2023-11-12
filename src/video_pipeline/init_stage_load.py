# External imports
from pypipeline.stages import IInitStage
from typing import Tuple

# Local imports
from .schemas.data_stage_load import DataStageLoad, DataStageOutputLoad
from .forward_stage_inference import ForwardStageInference


class InitStageLoad(IInitStage[DataStageLoad, DataStageOutputLoad, ForwardStageInference]):
    def compute(self) -> None:
        self._output = self.input.get_carry()

    def get_output(self) -> Tuple[ForwardStageInference, DataStageOutputLoad]:
        return ForwardStageInference(), DataStageOutputLoad(**self._output)