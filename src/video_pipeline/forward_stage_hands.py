# External imports
from pypipeline.stages import IForwardStage
from collections import defaultdict
from typing import Tuple
import numpy as np
import copy

# Local imports
from .forward_stage_joints import ForwardStageJoints
from .schemas.data_stage_inference import DataStageInference
from .schemas.data_stage_hands import DataStageHands


class ForwardStagehands(IForwardStage[DataStageInference, DataStageHands, ForwardStageJoints]):

    def __init__(self) -> None:
        super().__init__()

        self.kpt_map = {
            'carpal_base': 0,
            'carpal_thumb': 1,
            'thumb_base': 2,
            'thumb_middle': 3,
            'thumb_tip': 4,
            'index_base': 5,
            'index_middle1': 6,
            'index_middle2': 7,
            'index_tip': 8,
            'middle_base': 9,
            'middle_middle1': 10,
            'middle_middle2': 11,
            'middle_tip': 12,
            'ring_base': 13,
            'ring_middle1': 14,
            'ring_middle2': 15,
            'ring_tip': 16,
            'pinky_base': 17,
            'pinky_middle1': 18,
            'pinky_middle2': 19,
            'pinky_tip': 20
        }


    def convert_to_dictionary(self, kpts: dict) -> dict:
        kpts_dict = {}

        for key, k_index in self.kpt_map.items():
            kpts_dict[key] = np.array(kpts[k_index])

        kpts_dict['joints'] = np.array(list(self.kpt_map.keys()))

        return kpts_dict

    def count_digits(self, hand_kpt: np.array)-> int:
        tips = hand_kpt[[8, 12, 16, 20]] # 4 is thumb, we'll ignore for now
        mean_knuckle = np.mean(hand_kpt[[17, 13, 9, 5]], axis=0)
        raised_digits = 0
        for pt in tips:
            if pt[1] < mean_knuckle[1]:
                raised_digits += 1
        return raised_digits

    def compute(self) -> None:
        hand_kpts_detailed = []

        for kpts in self.input.keypoints_hands:
            kpts = np.array(kpts)
            kpts_dict = self.convert_to_dictionary(kpts)
            kpts_dict['digits_up'] = self.count_digits(kpts)

            hand_kpts_detailed.append(kpts_dict)

        self._output = {
            "hand_kpts_detailed": hand_kpts_detailed,
            **self.input.get_carry()
        }

    def get_output(self) -> Tuple[ForwardStageJoints, DataStageHands]:
        return ForwardStageJoints(), DataStageHands(**self._output)