# External imports
from pypipeline.stages import IForwardStage
from collections import defaultdict
from typing import Tuple
import numpy as np
import copy

# Local imports
from .terminal_stage import TerminalStage
from .schemas.data_stage_inference import DataStageInference
from .schemas.data_stage_joints import DataStageJoints


class ForwardStageJoints(IForwardStage[DataStageInference, DataStageJoints, TerminalStage]):

    def __init__(self) -> None:
        super().__init__()

        self.kpt_map = {
            'hips': 0,
            'hip_right': 1,
            'knee_right': 2,
            'feet_right': 3,
            'hip_left': 4,
            'knee_left': 5,
            'feet_left': 6,
            'spine_middle': 7,
            'spine_upper': 8,
            'neck': 9,
            'head_high': 10,
            'shoulder_left': 11,
            'elbow_left': 12,
            'hand_left': 13,
            'shoulder_right': 14,
            'elbow_right': 15,
            'hand_right': 16
        }

        self.skeletal_structure = {
            "leg_right": ["hips", "hip_right", "knee_right", "feet_right"],
            "leg_left": ["hips", "hip_left", "knee_left", "feet_left"],
            "spine": ["hips", "spine_middle", "spine_upper", "neck", "head_high"],
            "arm_left": ["spine_upper", "shoulder_left", "elbow_left", "hand_left"],
            "arm_right": ["spine_upper", "shoulder_right", "elbow_right", "hand_right"]
        }

        self.skeleton_connections = [
            ('hips', 'hip_right'),
            ('hips', 'hip_left'),
            ('hip_right', 'knee_right'),
            ('hip_left', 'knee_left'),
            ('knee_right', 'feet_right'),
            ('knee_left', 'feet_left'),
            ('hips', 'spine_middle'),
            ('spine_middle', 'spine_upper'),
            ('spine_upper', 'neck'),
            ('neck', 'head_high'),
            ('spine_upper', 'shoulder_left'),
            ('spine_upper', 'shoulder_right'),
            ('shoulder_left', 'elbow_left'),
            ('shoulder_right', 'elbow_right'),
            ('elbow_left', 'hand_left'),
            ('elbow_right', 'hand_right')
        ]


    def unit_vector(self, v1: np.ndarray) -> np.ndarray:
        return v1 / np.linalg.norm(v1)


    def angle_between(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)

        radians_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        degrees_angle = np.degrees(radians_angle)

        return (degrees_angle + 360) % 360


    def convert_to_dictionary(self, kpts: dict) -> dict:
        kpts_dict = {}

        for key, k_index in self.kpt_map.items():
            kpts_dict[key] = np.array(kpts[k_index])

        kpts_dict['joints'] = np.array(list(self.kpt_map.keys()))

        return kpts_dict


    def get_rotation_matrix(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
            uv_1, uv_2 = self.unit_vector(v1), self.unit_vector(v2)

            cos_t = np.sum(uv_1 * uv_2)
            sin_t = np.sqrt(np.sum(np.square(np.cross(uv_1,uv_2))))

            u = uv_1
            v = uv_2 - np.sum(uv_1 * uv_2)*uv_1
            v = v/np.sqrt(np.sum(np.square(v)))
            w = np.cross(uv_1, uv_2)
            w = w/np.sqrt(np.sum(np.square(w)))

            C = np.array([u, v, w])
            R_uvw = np.array([[cos_t, -sin_t, 0],
                            [sin_t, cos_t, 0],
                            [0, 0, 1]])

            R = C.T @ R_uvw @ C

            return R


    def decompose_rotation_matrix(self, R: np.ndarray) -> np.ndarray:
        thetaz = np.arctan2(-R[0,1], R[1,1])
        thetay = np.arctan2(-R[2,0], R[2,2])
        thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))

        return (
            (np.degrees(thetaz) + 360) % 360, 
            (np.degrees(thetay) + 360) % 360, 
            (np.degrees(thetax) + 360) % 360
        )


    def spine_hinge(self, kpts_dict: dict) -> tuple:
        v_1 = kpts_dict['hips']
        v_2 = kpts_dict['spine_middle']
        v_0 = np.array([v_1[0], v_1[1], v_2[2]])

        Z, X, Y = self.decompose_rotation_matrix(
            self.get_rotation_matrix((v_0 - v_1), (v_2 - v_1))
        )
        result = {'X': X, 'Y': Y, 'Z': Z}

        return result

    def compute_joint_angles(self, structure: dict, kpts_dict: dict) -> dict:
        kpts_dict['joint_angles'] = defaultdict(float)
        kpts_dict['joint_angles_decomposed'] = defaultdict(dict)

        for path_name, path_way in structure.items():
            joint_vertices = []
            path_way = copy.copy(path_way)

            while len(path_way) > 0:
                joint = path_way.pop(0)
                joint_vertices.append(joint)

                if len(joint_vertices) == 3:
                    v_0 = kpts_dict[joint_vertices[0]]
                    v_1 = kpts_dict[joint_vertices[1]]
                    v_2 = kpts_dict[joint_vertices[2]]

                    joint_angle = self.angle_between((v_0 - v_1), (v_2 - v_1))
                    kpts_dict['joint_angles'][joint_vertices[1]] = joint_angle

                    Z, X, Y = self.decompose_rotation_matrix(
                        self.get_rotation_matrix((v_0 - v_1), (v_2 - v_1))
                    )
                    kpts_dict['joint_angles_decomposed'][joint_vertices[1]]['X'] = X
                    kpts_dict['joint_angles_decomposed'][joint_vertices[1]]['Y'] = Z
                    kpts_dict['joint_angles_decomposed'][joint_vertices[1]]['Z'] = Y

                    joint_vertices.pop(0)
            
        kpts_dict['joint_angles_decomposed']['spine_hinge'] = self.spine_hinge(kpts_dict)

        return kpts_dict


    def compute(self) -> None:
        kpts_detailed = []

        for kpts in self.input.keypoints:
            kpts = np.array(kpts)
            kpts_dict = self.convert_to_dictionary(kpts)
            kpts_dict = self.compute_joint_angles(self.skeletal_structure, kpts_dict)

            kpts_detailed.append(kpts_dict)

        self._output = {
            "kpts_detailed": kpts_detailed,
            **self.input.get_carry()
        }


    def get_output(self) -> Tuple[TerminalStage, DataStageJoints]:
        return TerminalStage(), DataStageJoints(**self._output)