# External import
from pypipeline.controlers import BaseController
from rich import print
import argparse

# Local imports
from src.video_pipeline.init_stage_load import InitStageLoad
from src.video_pipeline.schemas.data_stage_load import DataStageLoad

def _parse_arguments():
    parser = argparse.ArgumentParser(description='Process some numbers.')
    parser.add_argument(
        '--input_video', 
        type=str, 
        help='Path to video of person performing exercise.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help='Folder which will store output graphs of joint angles.'
    )
    parser.add_argument(
        '--output_skeleton',
        type=str, 
        help='Folder which will store the video of the skeleton extracted.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_arguments()

    data = {
      "video_path": args.input_video,
      "video_output_path": args.output,
      "video_skeleton_path": args.output_skeleton
    }
    input_data = DataStageLoad(**data)
    controller = BaseController(input_data, InitStageLoad)
    output = controller.start()
