# External imports
from pypipeline.stages import ITerminalStage
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Tuple, List
from rich import print
import json
import matplotlib.animation as animation
import cv2


# Local imports
from .schemas.data_terminal_stage import DataStageTerminal
from .schemas.data_stage_classify import DataStageClassify
from src.utils.workout_rules import squat_rules, pullup_rules


class TerminalStage(ITerminalStage[DataStageClassify, DataStageTerminal]):

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


    def create_animation(self, data, intervals):
        cap = cv2.VideoCapture(self.input.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        num_joints = len(data.keys())

        # Calculate the number of columns needed
        num_cols = 2
        num_rows = (num_joints + 1) // num_cols + min(1, (num_joints + 1) % num_cols)

        # Create a figure and subplots for each joint and video
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

        # Flatten the axes array for easy iteration
        axs = axs.flatten()

        # Remove any extra subplot at the end
        if len(axs) > num_joints + 1:
            fig.delaxes(axs[-1])

        # Create lines to hold joint plots
        lines = [ax.plot([], [])[0] for ax in axs[:num_joints]]

        # Pre-draw rectangles for intervals in the joint subplot
        for i, (joint_name, joint_data) in enumerate(data.items()):
            for interval in intervals.get(joint_name, []):
                start, end = interval
                if 0 <= start < len(joint_data) and 0 <= end < len(joint_data):
                    rect = plt.Rectangle((start, 0), end - start, max(joint_data) * 1.1, color='red', alpha=0.3)
                    axs[i].add_patch(rect)

        # Iterate over each joint
        def update(frame):
            for i, (joint_name, joint_data) in enumerate(data.items()):
                # Set axis limits for the joint subplot
                axs[i].set_xlim(0, len(joint_data))
                axs[i].set_ylim(0, max(joint_data) * 1.1)

                x = np.arange(len(joint_data[:frame]))
                y = joint_data[:frame]
                lines[i].set_data(x, y)

            # Update the video subplot
            ret, frame_img = cap.read()
            if ret:
                axs[-1].clear()
                axs[-1].imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
                axs[-1].axis('off')

            return lines + [axs[-1]]

        # Create the animated plot
        ani = animation.FuncAnimation(fig, update, frames=len(next(iter(data.values()))), interval=1000 / fps, blit=True)
        ani.save("animates_plot.mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
    

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

        kr = np.array(self.simple_moving_average(joints_history[cyclic_pair[0]], 10))
        kl = np.array(self.simple_moving_average(joints_history[cyclic_pair[1]], 10))

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
                
            reps.append({"rep": i, "top": point, "bottom": next_bottom})

        return reps


    def validate_thresholding(self, joint: str, joints_history: dict, rep_dict: dict, good_reps: defaultdict) -> defaultdict:
        top_pos = joints_history[joint][rep_dict['top']]
        bottom_pos = joints_history[joint][rep_dict['bottom']]

        if self.ruleset[joint]["args"]['min'] >= top_pos:
            valid_top = True
            if 'valid' in good_reps[rep_dict['rep']]:
                valid_top = False if good_reps[rep_dict['rep']]['valid'] == False else True
        else:
            valid_top = False
        
        advice_top = {} if valid_top else {
            f'{joint}_top': {
                'position': 'top',
                'angle': top_pos,
                'expected': self.ruleset[joint]["args"]['min']
            },
            **(good_reps[rep_dict['rep']]['advice'] if 'advice' in good_reps[rep_dict['rep']] else {})
        }
        good_reps[rep_dict['rep']] = {
            'valid': valid_top,
            'advice': advice_top
        }
        
        if abs(self.ruleset[joint]["args"]['max'] - bottom_pos) <= 7:
            valid_bottom = True
            if 'valid' in good_reps[rep_dict['rep']]:
                valid_bottom = False if good_reps[rep_dict['rep']]['valid'] == False else True
        else:
            valid_bottom = False
        
        advice_bottom = {} if valid_bottom else {
            f'{joint}_bottom': {
                'position': 'bottom',
                'angle': bottom_pos,
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
            
        print(f"\n[bold magenta] You performed the following reps correctly:\n{json.dumps(dict(good_reps), indent=4)}")


    def compute(self) -> None:
        window_size = 10

        output_folder = self.input.video_output_path
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        print(self.input.workouts)

        #! Temporary - export the joints history as pickle
        # with open('joints_history.pickle', 'wb') as f:
        #     joints_history = defaultdict(list)

        #     for kpt_frame in np.array(self.input.kpts_detailed):
        #         for joint in kpt_frame['joint_angles']:
        #             joints_history[joint].append(
        #                 kpt_frame['joint_angles'][joint]
        #             )

        #     pickle.dump(joints_history, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # self._output = self.input.get_carry()
        # return

        print(self.input.hand_kpts_detailed)

        #! Testing the animation
        joints_history = defaultdict(list)

        for kpt_frame in np.array(self.input.kpts_detailed):
            for joint in kpt_frame['joint_angles']:
                joints_history[joint].append(
                    kpt_frame['joint_angles'][joint]
                )
        
        self.create_animation(
        {
            key: joints_history[key] for key in ["knee_right", "knee_left", "hip_left", "hip_right"]
        }, {
            'hip_right': [(100, 150), (300, 350)],
            'knee_left': [(50, 100), (199, 221)],
        })
        
        for workout in self.input.workouts:
            self.is_squat = workout["type"] == "squat"
            self.ruleset = squat_rules if self.is_squat else pullup_rules

            joints_history = defaultdict(list)

            for kpt_frame in np.array(self.input.kpts_detailed)[workout["start"]:workout["end"]]:
                for joint in kpt_frame['joint_angles']:
                    joints_history[joint].append(
                        kpt_frame['joint_angles'][joint]
                    )

            start_points, end_points = self.get_rep_points(joints_history)
            
            if len(end_points) > 0:
                rep_extremas = self.reparameterise_rep_points(start_points, end_points)
                pos_dependents = {
                    "hip_left": joints_history["hip_left"],
                    "hip_right": joints_history["hip_right"]
                } if self.is_squat else {
                    "knee_left": joints_history["knee_left"],
                    "knee_right": joints_history["knee_right"]
                }

                self.validate_reps_squat(rep_extremas, pos_dependents, joints_history)

            for joint_name in joints_history:
                plt.figure()

                joint_y = joints_history[joint_name]
                joint_sma_y = np.array(self.simple_moving_average(joint_y, window_size))

                # Plot joint angles
                plt.subplot()
                plt.figure().set_figwidth(12)
                plt.plot(joint_sma_y, label=joint_name)

                if joint_name in self.ruleset:
                    if 'max' in self.ruleset[joint_name]:
                        plt.axhline(
                            y=self.ruleset[joint_name]['max'],
                            color='g',
                            linestyle='--',
                            label=f'max={self.ruleset[joint_name]["max"]}'
                        )
                    
                    if 'min' in self.ruleset[joint_name]:
                        plt.axhline(
                            y=self.ruleset[joint_name]['min'],
                            color='r',
                            linestyle='--',
                            label=f'min={self.ruleset[joint_name]["min"]}'
                        )
                
                if len(end_points) > 0:
                    plt.scatter(
                        np.array(end_points),
                        np.array(joint_sma_y)[np.array(end_points)],
                        color='red', 
                        label='Rep. bottom'
                    )
                if len(start_points) > 0:
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
