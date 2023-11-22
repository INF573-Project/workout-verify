# External imports
from pypipeline.stages import ITerminalStage
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.signal import argrelextrema
import numpy as np
from collections import defaultdict

# Local imports
from .schemas.data_terminal_stage import DataStageTerminal
from .schemas.data_stage_joints import DataStageJoints


class TerminalStage(ITerminalStage[DataStageJoints, DataStageTerminal]):

    def __init__(self) -> None:
        super().__init__()

        self.cyclic_thresholds = {
            "knee": {
                "max": 145,
                "min": 90
            }
        }
    
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
        joint_rep_points = {}
        
        knee_history = (np.array(joints_history["knee_right"]) + np.array(joints_history["knee_left"])) / 2
        maximas = np.argwhere(knee_history >= self.cyclic_thresholds["knee"]["max"])
        minimas = np.argwhere(knee_history <= self.cyclic_thresholds["knee"]["min"])

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
        
        print(starting_points)
        print(end_points)


    def compute(self) -> None:
        window_size = 10

        output_folder = self.input.video_output_path
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        joints_history = defaultdict(list)

        for kpt_frame in self.input.kpts_detailed:
            for joint in kpt_frame['joint_angles']:
                joints_history[joint].append(kpt_frame['joint_angles'][joint])
            
            for joint in kpt_frame['joint_angles_decomposed']:
                joints_history[f'{joint}_X'].append(kpt_frame['joint_angles_decomposed'][joint]['X'])
                joints_history[f'{joint}_Y'].append(kpt_frame['joint_angles_decomposed'][joint]['Y'])
                joints_history[f'{joint}_Z'].append(kpt_frame['joint_angles_decomposed'][joint]['Z'])
        
        self.get_rep_points(joints_history)

        for joint_name in joints_history:
            plt.figure()

            joint_y = joints_history[joint_name]
            joint_sma_y = np.array(self.simple_moving_average(joint_y, window_size))

            # Plot joint angles
            plt.subplot(2, 1, 1)
            plt.plot(joint_sma_y, label=joint_name)

            plt.axhline(y=90, color='r', linestyle='-', label='y=90')
            plt.axhline(y=45, color='g', linestyle='--', label='y=45')

            plt.axhline(y=145, color='g', linestyle='--', label='y=145')
            plt.axhline(y=180, color='r', linestyle='-', label='y=180')

            local_maxima_indices = argrelextrema(joint_sma_y,np.greater)[0]
            local_minima_indices = argrelextrema(joint_sma_y, np.less)[0]

            plt.scatter(
                np.arange(len(joint_sma_y))[local_maxima_indices],
                np.array(joint_sma_y)[local_maxima_indices],
                color='red', 
                label='max'
            )
            plt.scatter(
                np.arange(len(joint_sma_y))[local_minima_indices], 
                np.array(joint_sma_y)[local_minima_indices],
                color='green', 
                label='min'
            )

            plt.xlabel('Frame Index')
            plt.ylabel('Angle (degrees)')
            plt.legend()

            # Plot decomposed joint angles
            plt.subplot(2, 1, 2)
            for axis in ['X', 'Y', 'Z']:
                decomposed_key = f'{joint}_{axis}'
                plt.plot(
                    self.simple_moving_average(joints_history[decomposed_key], window_size), 
                    label=decomposed_key
                )
            plt.title(f'Decomposed Joint Angles: {joint_name}')
            plt.xlabel('Frame Index')
            plt.ylabel('Angle (degrees)')
            plt.legend()

            # Save the plot
            plt.tight_layout()
            output_path = Path(output_folder) / f'{joint_name}_plot.png'
            plt.savefig(output_path)
            plt.close()

        self._output = self.input.get_carry()

    def get_output(self) -> DataStageTerminal:
        return DataStageTerminal(**self._output)
