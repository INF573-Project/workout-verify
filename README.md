# Workout Verify

**Authors: Patrick Tourniaire & Justin Regef**

This is a project for automatically detecting if a person in a video is performing a squat or a pull-up somewhere in the video. Which is used to then give local feedback on the execution of the exercise(s) performed.


## Setup
First you have to install the poetry dependencies and implicitly setup a venv with those deps.

```Bash
$ pip install poetry
$ poetry install
```

Then you have to set8up MMPose to be able to load and infer on the pose estimation models.

```Bash
$ poetry run mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
```

## Running

After running the above setup, you should be able to simply provide an input video to the `main.py` file using the `--input_video` argument.

**Note:** the processing time for extracting the pose skeletons can take a long time, therefore, we also have some example data of us performing exercises under `data/examples/*` which has cached pickle files for the computed skeletons.

```Bash
$ poetry run python main.py --input_video <path_to_video>
```

If you want to run based on a cached sequence of skeletons then you can simply run this command, and refer to a specific pickle file which you want to test.

```Bash
$ poetry run python main.py --input_video cache/<video_name>.pickle
```

## Issues & Contact

If you experience any issues when running this project, feel free to contact us:

Justin Regef: **justin.regef@polytechnique.edu**

Patrick Tourniaire: **patrick.tourniaire@polytechnique.edu**

## Side Note 

This project relies on the `pypipeline` package which is to publicly available yet but was developed by Patrick Tourniaire, and is available in the `deps/` folder as a wheel file. However, running `poetry install` will take care of this, but if you decide not to use poetry then it is important to also install this package.
