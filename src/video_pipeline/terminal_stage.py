# External imports
from pypipeline.stages import ITerminalStage
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.signal import argrelextrema
import numpy as np
from collections import defaultdict
from typing import Tuple, List
from rich import print
import json

# Local imports
from .schemas.data_terminal_stage import DataStageTerminal
from .schemas.data_stage_joints import DataStageJoints
from src.utils.workout_rules import squat_rules


class TerminalStage(ITerminalStage[DataStageJoints, DataStageTerminal]):

    def __init__(self) -> None:
        super().__init__()

    def simple_moving_average(self, data, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        moving_averages = []
        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size]
            average = sum(window) / window_size
            moving_averages.append(average)

        return moving_averages


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
    

    def get_rep_points(self, joints_history: dict) -> dict:
        kr = np.array(self.simple_moving_average(joints_history["knee_right"], 10))
        kl = np.array(self.simple_moving_average(joints_history["knee_left"], 10))

        knee_history = (kr + kl) / 2
        maximas = np.argwhere(knee_history >= squat_rules["knee_right"]["max"])
        minimas = np.argwhere(knee_history <= squat_rules["knee_right"]["min"])

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
                
            reps.append({"rep": i, "top": point, "bottom": next_bottom})

        return reps
    
    def validate_reps_squat(self, cyclic_extremas: List[dict], pos_dep_joints: dict):
        good_reps = defaultdict(dict)
        
        for rep_dict in cyclic_extremas:
            if 'bottom' not in rep_dict:
                good_reps[rep_dict['rep']] = {
                    'valid': False,
                    'advice': {
                        'knees_bottom': {
                            'angle': 21,
                            'expected': squat_rules["knee_right"]["min"]
                        }
                    }
                }
            else:
                for joint in pos_dep_joints:
                    top_pos = pos_dep_joints[joint][rep_dict['top']]
                    bottom_pos = pos_dep_joints[joint][rep_dict['bottom']]

                    if squat_rules[joint]['min'] <= top_pos:
                        valid_top = True
                        if 'valid' in good_reps[rep_dict['rep']]:
                            valid_top = False if good_reps[rep_dict['rep']]['valid'] == False else True
                    else:
                        valid_top = False
                    
                    advice_top = {} if valid_top else {
                        f'{joint}_top': {
                            'position': 'top',
                            'angle': top_pos,
                            'expected': squat_rules[joint]['min']
                        },
                        **(good_reps[rep_dict['rep']]['advice'] if 'advice' in good_reps[rep_dict['rep']] else {})
                    }
                    good_reps[rep_dict['rep']] = {
                        'valid': valid_top,
                        'advice': advice_top
                    }
                    
                    if abs(squat_rules[joint]['max'] - bottom_pos) <= 7:
                        valid_bottom = True
                        if 'valid' in good_reps[rep_dict['rep']]:
                            valid_bottom = False if good_reps[rep_dict['rep']]['valid'] == False else True
                    else:
                        valid_bottom = False
                    
                    advice_bottom = {} if valid_bottom else {
                        f'{joint}_bottom': {
                            'position': 'bottom',
                            'angle': bottom_pos,
                            'expected': squat_rules[joint]['max']
                        },
                        **(good_reps[rep_dict['rep']]['advice'] if 'advice' in good_reps[rep_dict['rep']] else {})
                    }
                    good_reps[rep_dict['rep']] = {
                        'valid': valid_bottom,
                        'advice': advice_bottom
                    }
            
        
        print(f"\n[bold magenta] You performed the following reps correctly:\n{json.dumps(dict(good_reps), indent=4)}")


    def compute(self) -> None:
        window_size = 10

        output_folder = self.input.video_output_path
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        joints_history = defaultdict(list)

        for kpt_frame in self.input.kpts_detailed:
            for joint in kpt_frame['joint_angles']:
                joints_history[joint].append(kpt_frame['joint_angles'][joint])
        
        start_points, end_points = self.get_rep_points(joints_history)
        rep_extremas = self.reparameterise_rep_points(start_points, end_points)
        self.validate_reps_squat(
            rep_extremas,
            {
                "hip_left": joints_history["hip_left"],
                "hip_right": joints_history["hip_right"]
            }
        )

        for joint_name in joints_history:
            plt.figure()

            joint_y = joints_history[joint_name]
            joint_sma_y = np.array(self.simple_moving_average(joint_y, window_size))

            # Plot joint angles
            plt.subplot()
            plt.figure().set_figwidth(12)
            plt.plot(joint_sma_y, label=joint_name)

            if joint_name in squat_rules:
                if 'max' in squat_rules[joint_name]:
                    plt.axhline(
                        y=squat_rules[joint_name]['max'],
                        color='g',
                        linestyle='--',
                        label=f'max={squat_rules[joint_name]["max"]}'
                    )
                
                if 'min' in squat_rules[joint_name]:
                    plt.axhline(
                        y=squat_rules[joint_name]['min'],
                        color='r',
                        linestyle='--',
                        label=f'min={squat_rules[joint_name]["min"]}'
                    )

            plt.scatter(
                np.array(end_points),
                np.array(joint_sma_y)[np.array(end_points)],
                color='red', 
                label='Rep. bottom'
            )
            plt.scatter(
                np.array(start_points),
                np.array(joint_sma_y)[np.array(start_points)],
                color='green', 
                label='Rep. top'
            )

            plt.xlabel('Frame Index')
            plt.ylabel('Angle (degrees)')
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.15),
                fancybox=True,
                ncol=7
            )

            # Save the plot
            plt.tight_layout()
            output_path = Path(output_folder) / f'{joint_name}_plot.png'
            plt.savefig(output_path)
            plt.close()

        self._output = self.input.get_carry()

    def get_output(self) -> DataStageTerminal:
        return DataStageTerminal(**self._output)
