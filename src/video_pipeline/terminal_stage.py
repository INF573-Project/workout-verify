# External imports
from pypipeline.stages import ITerminalStage
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from rich import print
import matplotlib.animation as animation
import cv2


# Local imports
from .schemas.data_terminal_stage import DataStageTerminal
from .schemas.data_stage_advice import DataStageAdvice
from src.utils.utils import simple_moving_average


class TerminalStage(ITerminalStage[DataStageAdvice, DataStageTerminal]):

    def __init__(self) -> None:
        super().__init__()
    

    def _plot_joint_plots(self):
        window_size = 10
        joints_history = self.input.joints_history
        
        for joint_name in joints_history:
                plt.figure()

                joint_y = joints_history[joint_name]
                joint_sma_y = np.array(simple_moving_average(joint_y, window_size))

                plt.subplot()
                plt.figure().set_figwidth(12)
                plt.plot(joint_sma_y, label=joint_name)

                plt.xlabel('Frame Index')
                plt.ylabel('Angle (degrees)')
                plt.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.15),
                    fancybox=True,
                    ncol=7
                )

                plt.tight_layout()
                output_path = Path(self.input.video_output_path) / f'{joint_name}_plot.png'
                plt.savefig(output_path)
                plt.close()
        

    def _create_animation(self, data, intervals):
        cap = cv2.VideoCapture(self.input.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        num_joints = len(data.keys())

        num_cols = 2
        num_rows = num_joints // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
        axs = axs.flatten()

        # Create lines to hold joint plots
        lines = [ax.plot([], [])[0] for ax in axs[:num_joints]]

        # Pre-draw rectangles for intervals in the joint subplot
        for i, (joint_name, joint_data) in enumerate(data.items()):
            for interval in intervals.get(joint_name, []):
                start, end = interval
                if 0 <= start < len(joint_data) and 0 <= end < len(joint_data):
                    rect = plt.Rectangle((start, min(joint_data)-10), end - start, 180, color='red', alpha=0.3)
                    axs[i].add_patch(rect)

        for i, joint_name in enumerate(data.keys()):
            axs[i].set_title(" ".join([word.capitalize() for word in joint_name.split("_")]))

        # Iterate over each joint
        def update(frame):
            for i, (_, joint_data) in enumerate(data.items()):
                # Set axis limits for the joint subplot
                axs[i].set_xlim(0, len(joint_data))

                axs[i].set_ylim(min(joint_data) - 10, max(joint_data) + 10)

                x = np.arange(len(joint_data[:frame]))
                y = joint_data[:frame]
                lines[i].set_data(x, y)

            # Update the video subplot
            # ret, frame_img = cap.read()
            # if ret:
            #     axs[-1].clear()
            #     axs[-1].imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            #     axs[-1].axis('off')

            return lines + [axs[-1]]

        # Create the animated plot
        ani = animation.FuncAnimation(fig, update, frames=len(list(data.values())[0].tolist()), interval=1000 / fps, blit=True)
        ani.save("animates_plot.mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
    


    def compute(self) -> None:
        output_folder = self.input.video_output_path
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        joints_history = self.input.joints_history
        
        knee_joints = (np.array(joints_history["knee_right"]) + np.array(joints_history["knee_left"])) / 2
        hip_joints = (np.array(joints_history["hip_right"]) + np.array(joints_history["hip_left"])) / 2
        elbow_joints = (np.array(joints_history["elbow_right"]) + np.array(joints_history["elbow_left"])) / 2
        direct_plots = ["spine_middle"]

        markers = []

        for advice in self.input.workout_advice[0]['advice']:
            if not self.input.workout_advice[0]['advice'][advice]['valid']:
                rep_dict = self.input.workout_rep_extrema['squat'][0][advice]
                start = rep_dict["top"]
                end = start + 50

                if 'bottom' in rep_dict:
                    end = rep_dict["bottom"]
                
                markers.append((start, end))

        self._create_animation(
        {
            "knee_joints": knee_joints,
            "hip_joints": hip_joints,
            "elbow_joints": elbow_joints,
            **{key: self.input.joints_history[key] for key in direct_plots},
        }, {
            'knee_joints': markers,
        })
        self._plot_joint_plots()

        self._output = self.input.get_carry()

    def get_output(self) -> DataStageTerminal:
        return DataStageTerminal(**self._output)
