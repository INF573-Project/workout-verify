# External imports
from pypipeline.stages import IForwardStage
from collections import defaultdict
from typing import Tuple
import numpy as np

# Local imports
from .terminal_stage import TerminalStage
from .schemas.data_stage_inference import DataStageInference
from .schemas.data_stage_joints import DataStageJoints


class ForwardStageJoints(IForwardStage[DataStageInference, DataStageJoints, TerminalStage]):

    def __init__(self) -> None:
        super().__init__()

        self.kpt_dict = {0: 'hips',
            1: 'righthip',
            2: 'knee_right',
            3: 'feet_right',
            4: 'lefthip',
            5: 'knee_left',
            6: 'feet_left',
            7: 'spine_middle',
            8: 'spine_upper',
            9: 'neck',
            10: 'head_high',
            11: 'shoulder_left',
            12: 'elbow_left',
            13: 'hand_left',
            14: 'shoulder_right',
            15: 'elbow_right',
            16: 'hand_right'}

        self.reversed_kpt_dict = {
            value: key for key, value in self.kpt_dict.items()
        }
    
    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)

        radians_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        degrees_angle = np.degrees(radians_angle)

        return (degrees_angle + 360) % 360

    def Get_R(self, A,B):
        uA = A/np.sqrt(np.sum(np.square(A)))
        uB = B/np.sqrt(np.sum(np.square(B)))

        cos_t = np.sum(uA * uB)
        sin_t = np.sqrt(np.sum(np.square(np.cross(uA,uB)))) #magnitude

        u = uA
        v = uB - np.sum(uA * uB)*uA
        v = v/np.sqrt(np.sum(np.square(v)))
        w = np.cross(uA, uB)
        w = w/np.sqrt(np.sum(np.square(w)))

        C = np.array([u, v, w])
        R_uvw = np.array([[cos_t, -sin_t, 0],
                        [sin_t, cos_t, 0],
                        [0, 0, 1]])

        R = C.T @ R_uvw @ C
        return R

    def Decompose_R_ZXY(self, R):
        thetaz = np.arctan2(-R[0,1], R[1,1])
        thetay = np.arctan2(-R[2,0], R[2,2])
        thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))

        return np.degrees(thetaz), np.degrees(thetay), np.degrees(thetax)

    def compute(self) -> None:
        joint_angles_decomposed = defaultdict(list)
        joint_angles = defaultdict(list)

        for kpt in self.input.keypoints:
            a = kpt[4]
            b = kpt[5]
            c = kpt[6]

            ba = a - b
            bc = c - b

            R1 = self.Get_R(ba, bc)
            z1, y1, x1 = self.Decompose_R_ZXY(R1)
            joint_angles_decomposed['knee_left_x'].append(x1)
            joint_angles_decomposed['knee_left_y'].append(y1)
            joint_angles_decomposed['knee_left_z'].append(z1)
            joint_angles['knee_left'].append(self.angle_between(ba, bc))

            a = kpt[1]
            b = kpt[2]
            c = kpt[3]

            ba = a - b
            bc = c - b

            R1 = self.Get_R(ba, bc)
            z1, y1, x1 = self.Decompose_R_ZXY(R1)
            joint_angles_decomposed['knee_right_x'].append(x1)
            joint_angles_decomposed['knee_right_y'].append(y1)
            joint_angles_decomposed['knee_right_z'].append(z1)
            joint_angles['knee_right'].append(self.angle_between(ba, bc))

        self._output = {
            "joint_angles_decomposed": joint_angles_decomposed,
            "joint_angles": joint_angles,
            **self.input.get_carry()
        }

    def get_output(self) -> Tuple[TerminalStage, DataStageJoints]:
        return TerminalStage(), DataStageJoints(**self._output)