# External imports
from pypipeline.stages import ITerminalStage
from matplotlib import pyplot as plt
from pathlib import Path

# Local imports
from .schemas.data_terminal_stage import DataStageTerminal
from .schemas.data_stage_joints import DataStageJoints


class TerminalStage(ITerminalStage[DataStageJoints, DataStageTerminal]):
    
    def simple_moving_average(self, data, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        moving_averages = []
        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size]
            average = sum(window) / window_size
            moving_averages.append(average)

        return moving_averages

    def compute(self) -> None:
        window_size = 10

        output_folder = "data/examples/output"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        for joint_name in self.input.joint_angles:
            plt.figure()

            # Plot joint angles
            plt.subplot(2, 1, 1)
            plt.plot(
                self.simple_moving_average(
                    self.input.joint_angles[joint_name], window_size
                ),
                label=joint_name
            )
            plt.title(f'Joint Angle: {joint_name}')
            plt.xlabel('Frame Index')
            plt.ylabel('Angle (degrees)')
            plt.legend()

            # Plot decomposed joint angles
            plt.subplot(2, 1, 2)
            for axis in ['x', 'y', 'z']:
                decomposed_key = f'{joint_name}_{axis}'
                plt.plot(
                    self.simple_moving_average(
                        self.input.joint_angles_decomposed[decomposed_key], window_size
                    ), 
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
