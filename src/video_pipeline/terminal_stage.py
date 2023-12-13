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

                # Plot joint angles
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

                # Save the plot
                plt.tight_layout()
                output_path = Path(self.input.video_output_path) / f'{joint_name}_plot.png'
                plt.savefig(output_path)
                plt.close()
        

    def _create_animation(self, data, intervals):
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
    


    def compute(self) -> None:
        output_folder = self.input.video_output_path
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        print(self.input.workout_advice)
        
        # self._create_animation(
        # {
        #     key: self.input.joints_history[key] for key in ["knee_right", "knee_left", "hip_left", "hip_right"]
        # }, {
        #     'hip_right': [(100, 150), (300, 350)],
        #     'knee_left': [(50, 100), (199, 221)],
        # })
        self._plot_joint_plots()

        self._output = self.input.get_carry()

    def get_output(self) -> DataStageTerminal:
        return DataStageTerminal(**self._output)
