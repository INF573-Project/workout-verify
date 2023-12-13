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

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_arguments()
    file_name = args.input_video.split("/")[-1].split(".")[0]

    data = {
      "video_path": args.input_video,
      "file_name": file_name,
      "video_output_path": f"data/output/{file_name}/",
      "video_skeleton_path": f"data/output_skeleton/{file_name}/"
    }
    input_data = DataStageLoad(**data)
    controller = BaseController(input_data, InitStageLoad)
    controller.discover()
    output, run_id = controller.start()

    artifacts = controller.get_artifacts(run_id)
