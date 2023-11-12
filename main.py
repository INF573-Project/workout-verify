# External import
from pypipeline.controlers import BaseController
from rich import print

# Local imports
from src.video_pipeline.init_stage_load import InitStageLoad
from src.video_pipeline.schemas.data_stage_load import DataStageLoad

data = {
  "video_path": "./data/examples/squat_video.mp4",
  "video_output_path": "./data/output"
}
input_data = DataStageLoad(**data)
controller = BaseController(input_data, InitStageLoad)
output = controller.start()

print(output)
