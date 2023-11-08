# workout-verify

## Setup
First you have to install the poetry dependencies and implicitly setup a vnenv with those deps.

```Bash
$ poetry install
```

Then you have to setup MMPose to be able to load and infer on the pose estimation models.

```Bash
$ poetry run mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
```
