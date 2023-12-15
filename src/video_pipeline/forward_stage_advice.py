# External imports
from pypipeline.stages import IForwardStage
from collections import defaultdict
from typing import Tuple, List
import numpy as np
from pathlib import Path
import pickle
from rich.console import Console
from rich.table import Table

# Local imports
from .terminal_stage import TerminalStage
from .schemas.data_stage_classify import DataStageClassify
from .schemas.data_stage_advice import DataStageAdvice

from src.models.workout_classifier import WorkoutClassifier
from src.utils.workout_rules import squat_rules, pullup_rules
from src.utils.utils import simple_moving_average


class ForwardStageAdvice(IForwardStage[DataStageClassify, DataStageAdvice, TerminalStage]):

    def __init__(self) -> None:
        super().__init__()
  

    def _build_joints_history(self):
        joints_history = defaultdict(list)

        for kpt_frame in self.input.kpts_detailed:
            for joint in kpt_frame['joint_angles']:
                joints_history[joint].append(kpt_frame['joint_angles'][joint])
        
        return joints_history


    def group_extremas(self, extrema_indecies: np.ndarray) -> dict:
        groups = defaultdict(list)
        group_index = 0
        group_thres = 5

        for i in range(len(extrema_indecies)):
            if i - 1 < 0: groups[group_index].append(extrema_indecies[i][0])
            diff = extrema_indecies[i] - extrema_indecies[i - 1]

            if diff <= group_thres:
                groups[group_index].append(extrema_indecies[i][0])
            else:
                group_index += 1
        
        return groups
  

    def get_cyclic_joints(self) -> list:
        cyclic_joints = []

        for joint in self.ruleset:
            if self.ruleset[joint]["cyclic_joint"]:
                cyclic_joints.append(joint)
        
        return cyclic_joints


    def get_delta_thresholding_joints(self) -> list:
        delta_joints = []

        for joint in self.ruleset:
            if self.ruleset[joint]["type"] == "delta_thresholding":
                delta_joints.append(joint)
        
        return delta_joints


    def get_rep_points(self, joints_history: dict) -> dict:
        cyclic_pair = self.get_cyclic_joints()
        if len(cyclic_pair) != 2:
            raise ValueError(f"""There should be exactly two cyclic joints defined in the ruleset
                             for a movement, found the following: {cyclic_pair}""")

        kr = np.array(simple_moving_average(joints_history[cyclic_pair[0]], 10))
        kl = np.array(simple_moving_average(joints_history[cyclic_pair[1]], 10))

        knee_history = (kr + kl) / 2
        maximas = np.argwhere(knee_history >= self.ruleset[cyclic_pair[0]]["args"]["max"])
        minimas = np.argwhere(knee_history <= self.ruleset[cyclic_pair[1]]["args"]["min"])

        starting_points = []
        end_points = []

        maxima_groups = self.group_extremas(maximas)
        minima_groups = self.group_extremas(minimas)

        for i in maxima_groups:
            indecies = maxima_groups[i]
            starting_points.append(indecies[np.argmax(knee_history[indecies])])

        for i in minima_groups:
            indecies = minima_groups[i]
            end_points.append(indecies[np.argmin(knee_history[indecies])])
        
        return starting_points, end_points

    
    def reparameterise_rep_points(self, start_points: list, end_points: list) -> List[Tuple[str, int, int]]:
        reps = []

        for i, point in enumerate(start_points):
            if i + 1 >= len(start_points):
                break

            next_start = start_points[i + 1]
            next_bottom = None

            for bottom in end_points:
                if bottom > point and bottom < next_start:
                    next_bottom = bottom
            
            if next_bottom is None:
                reps.append({"rep": i, "top": point})
                continue
                
            reps.append({"rep": i, "top": point, "bottom": next_bottom, "invalidated": False})

        return reps
    

    def validate_chin_pos(self, rep_extremas: List[dict]) -> List[dict]:
        filtered_extremas = []
        
        for extrema in rep_extremas:
            if "bottom" in extrema:
                chin_top_z = self.input.kpts_detailed[extrema["top"]]["head_high"][2]
                handr_top_z = self.input.kpts_detailed[extrema["top"]]["hand_right"][2]
                handl_top_z = self.input.kpts_detailed[extrema["top"]]["hand_left"][2]
                hand_top_z = (handr_top_z + handl_top_z) / 2
                
                extrema["invalidated"] = (hand_top_z - chin_top_z) < 0
                filtered_extremas.append(extrema)
        
        return filtered_extremas


    def validate_thresholding(self, joint: str, joints_history: dict, rep_dict: dict, good_reps: defaultdict) -> defaultdict:
        top_pos = joints_history[joint][rep_dict['top']]
        bottom_pos = joints_history[joint][rep_dict['bottom']]

        if abs(self.ruleset[joint]["args"]['min'] - abs(top_pos)) <= 8:
            valid_top = True
            if 'valid' in good_reps[rep_dict['rep']]:
                valid_top = False if good_reps[rep_dict['rep']]['valid'] == False else True
        else:
            valid_top = False
        
        advice_top = {} if valid_top else {
            f'{joint}_top': {
                'position': 'top',
                'angle': abs(top_pos),
                'expected': self.ruleset[joint]["args"]['min']
            },
            **(good_reps[rep_dict['rep']]['advice'] if 'advice' in good_reps[rep_dict['rep']] else {})
        }
        good_reps[rep_dict['rep']] = {
            'valid': valid_top,
            'advice': advice_top
        }
        
        if abs(self.ruleset[joint]["args"]['max'] - abs(bottom_pos)) <= 8:
            valid_bottom = True
            if 'valid' in good_reps[rep_dict['rep']]:
                valid_bottom = False if good_reps[rep_dict['rep']]['valid'] == False else True
        else:
            valid_bottom = False
        
        advice_bottom = {} if valid_bottom else {
            f'{joint}_bottom': {
                'position': 'bottom',
                'angle': abs(bottom_pos),
                'expected': self.ruleset[joint]["args"]['max']
            },
            **(good_reps[rep_dict['rep']]['advice'] if 'advice' in good_reps[rep_dict['rep']] else {})
        }
        good_reps[rep_dict['rep']] = {
            'valid': valid_bottom,
            'advice': advice_bottom
        }

        return good_reps
    
    
    def validate_delta_thresholding(self, joint: str, joints_history: dict, rep_dict: dict, good_reps: defaultdict) -> defaultdict:
        deltas = []

        for i, angle in enumerate(joints_history[joint]):
            if i == 0: continue
            deltas.append(angle - joints_history[joint][i-1])

        deltas = np.array(deltas)
        above_max = np.argwhere(deltas > self.ruleset[joint]["args"]["max"])
        above_min = np.argwhere(deltas < self.ruleset[joint]["args"]["min"])

        if len(above_max) > 0 or len(above_min) > 0:
            good_reps[rep_dict['rep']] = {
                "valid": False,
                "advice": {
                    joint: {
                        "expected": "Low variation in leg momentum",
                        "actual": "Exceeded delta threshold"
                    }
                }
            }
        else:
            good_reps[rep_dict['rep']] = {
                "valid": True,
                "advice": {}
            }

        return good_reps

    
    def validate_reps_squat(self, cyclic_extremas: List[dict], pos_dep_joints: dict, joints_history: dict):
        good_reps = defaultdict(dict)
        
        for rep_dict in cyclic_extremas:
            if 'bottom' not in rep_dict:
                if self.is_squat:
                    good_reps[rep_dict['rep']] = {
                        'valid': False,
                        'advice': {
                            'knees_bottom': {
                                'expected': self.ruleset["knee_right"]["args"]["min"]
                            }
                        }
                    }
                else:
                    good_reps[rep_dict['rep']] = {
                        'valid': False,
                        'advice': {
                            'elbows_top': {
                                'expected': self.ruleset["elbow_right"]["args"]["min"]
                            }
                        }
                    }
            elif rep_dict["invalidated"]:
                good_reps[rep_dict['rep']] = {
                    'valid': False,
                    'advice': {
                        'knees_bottom': {
                            'expected': "Head above hand at top"
                        }
                    }
                }
            else:
                if len(pos_dep_joints) > 0:
                    for joint in pos_dep_joints:
                        if self.ruleset[joint]["type"] == "thresholding":
                            good_reps = self.validate_thresholding(joint, joints_history, rep_dict, good_reps)
                        if self.ruleset[joint]["type"] == "delta_thresholding":
                            good_reps = self.validate_delta_thresholding(joint, joints_history, rep_dict, good_reps)
                else:
                    good_reps[rep_dict['rep']] = {
                        'valid': True,
                        'advice': {}
                    }
            
        return good_reps
    

    def _pretty_print(self, advice: dict, extremas: List[dict], w_type: str) -> None:
        table = Table(title=f"Workout Type: {w_type.capitalize()}")

        table.add_column("Rep. Number", justify="right", style="cyan", no_wrap=True)
        table.add_column("Valid Rep.", style="magenta")
        table.add_column("Frames", style="magenta")
        table.add_column("Advice", justify="right", style="green")

        for rep_num in advice:
            frames = f"{extremas[rep_num]['top']} - {extremas[rep_num]['bottom']}" if 'bottom' in extremas[rep_num] else f"{extremas[rep_num]['top']} - _"

            feedback = ""

            for joint in advice[rep_num]['advice']:
                if 'angle' not in advice[rep_num]['advice'][joint]:
                    feedback += "You did not go at parallel or beyond - go lower! \n"
                    continue
    
                actual = advice[rep_num]['advice'][joint]['angle']
                expected = advice[rep_num]['advice'][joint]['expected']

                if abs(expected - actual) > 10:
                    feedback += f"Gap between expected ({expected}°) and actual ({actual:.0f}°) for {joint} \n"

            if feedback == "": feedback = "Perfectly executed!"

            table.add_row(
                str(rep_num), 
                str(advice[rep_num]['valid']),
                frames,
                feedback
            )

        console = Console()
        console.print(table)
    

    def compute(self) -> None:
        joints_history = self._build_joints_history()
        workout_advice = []

        joints_history = defaultdict(list)

        for kpt_frame in np.array(self.input.kpts_detailed):
            for joint in kpt_frame['joint_angles']:
                joints_history[joint].append(
                    kpt_frame['joint_angles'][joint]
                )
        
        joint_histories_path = Path("./cache/joint_histories") / f'{self.input.file_name}_ktps_joints.pickle'

        with open(joint_histories_path, 'wb') as f:
            pickle.dump(joints_history, f, protocol=pickle.HIGHEST_PROTOCOL)

        workout_rep_extrema = defaultdict(list)

        for workout in self.input.workouts:
            if workout["type"] == "random": continue
            
            self.is_squat = workout["type"] == "squat"
            self.ruleset = squat_rules if self.is_squat else pullup_rules

            joints_history = defaultdict(list)

            for kpt_frame in np.array(self.input.kpts_detailed)[workout["start"]:workout["end"]]:
                for joint in kpt_frame['joint_angles']:
                    joints_history[joint].append(
                        kpt_frame['joint_angles'][joint]
                    )

            with open(joint_histories_path, 'wb') as f:
                pickle.dump(joints_history, f, protocol=pickle.HIGHEST_PROTOCOL)

            start_points, end_points = self.get_rep_points(joints_history)

            if len(end_points) > 0:
                rep_extremas = self.reparameterise_rep_points(start_points, end_points)
                
                if workout["type"] == "pullup":
                    rep_extremas = self.validate_chin_pos(rep_extremas)
                
                workout_rep_extrema[workout["type"]].append(rep_extremas)

                pos_dependents = {
                    "hip_left": joints_history["hip_left"],
                    "hip_right": joints_history["hip_right"]
                } if self.is_squat else {
                    "knee_left": joints_history["knee_left"],
                    "knee_right": joints_history["knee_right"]
                }

                advice = self.validate_reps_squat(rep_extremas, pos_dependents, joints_history)

                self._pretty_print(advice, rep_extremas, workout["type"])

                workout_advice.append({
                    **workout,
                    "advice": advice
                })

        
        self._output = {
            "joints_history": joints_history,
            "workout_advice": workout_advice,
            "workout_rep_extrema": workout_rep_extrema,
            **self.input.get_carry()
        }


    def get_output(self) -> Tuple[TerminalStage, DataStageAdvice]:
        return TerminalStage(), DataStageAdvice(**self._output)