# External imports
from mmpose.apis import MMPoseInferencer
from pypipeline.stages import IForwardStage
from typing import Tuple
import numpy as np

# Local imports
from .forward_stage_joints import ForwardStageJoints
from .schemas.data_stage_load import DataStageOutputLoad
from .schemas.data_stage_inference import DataStageInference


class ForwardStageInference(IForwardStage[DataStageOutputLoad, DataStageInference, ForwardStageJoints]):
    def compute(self) -> None:
        inferencer = MMPoseInferencer(pose3d='human3d')
        result_generator = inferencer(self.input.video_path)
        kpts = []

        for result in result_generator:
            kpts.append(np.array(result['predictions'][0][0]['keypoints']))
        
        self._output = {"keypoints": kpts, **self.input.get_carry()}

    def get_output(self) -> Tuple[ForwardStageJoints, DataStageInference]:
        return ForwardStageJoints(), DataStageInference(**self._output)