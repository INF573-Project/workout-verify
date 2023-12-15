# External imports
from pypipeline.stages import IForwardStage
from collections import defaultdict
from typing import Tuple, List
from rich import print

# Local imports
from .forward_stage_advice import ForwardStageAdvice
from .schemas.data_stage_joints import DataStageJoints
from .schemas.data_stage_classify import DataStageClassify

from src.models.workout_classifier import WorkoutClassifier
from src.utils.joint_features import JointFeatures


class ForwardStageClassify(IForwardStage[DataStageJoints, DataStageClassify, ForwardStageAdvice]):

    def __init__(self) -> None:
        super().__init__()
        
        self.classifier = WorkoutClassifier('src/models/sliding_window_classifier.pkl')
    

    def _build_joints_history(self):
        joints_history = defaultdict(list)

        for kpt_frame in self.input.kpts_detailed:
            for joint in kpt_frame['joint_angles']:
                joints_history[joint].append(kpt_frame['joint_angles'][joint])
        
        return joints_history


    def _merge_consecutive(self, groups: List[dict]) -> List[dict]:
        merged_data = []
        current = None

        for group in groups:
            if current is None or current['type'] != group['type']:
                current = group.copy()
                merged_data.append(current)
            else:
                current['end'] = group['end']

        return merged_data


    def _filter_invalid(self, groups: List[dict]) -> List[dict]:
        return [group for group in groups if group["start"] < group["end"]]


    def _update_times(self, groups: List[dict]) -> List[dict]:
        random_indices = [i for i, d in enumerate(groups) if d['type'] == 'random']

        for idx in random_indices:
            if idx > 0 and groups[idx - 1]['type'] != 'random':
                groups[idx]['start'] = groups[idx - 1]['end'] + 1

            if idx < len(groups) - 1 and groups[idx + 1]['type'] != 'random':
                groups[idx]['end'] = groups[idx + 1]['start'] - 1

        return groups


    def _parse_sliding_windows(self, sliding_windows: List[int]) -> List[dict]:
        workout_types = ["pullup", "squat"]
        groups = []
        start = 0
        current_num = sliding_windows[0]
        joint_features = JointFeatures()

        for i in range(1, len(sliding_windows)):
            if sliding_windows[i] != current_num:
                groups.append({
                    'type': workout_types[current_num],
                    'start': start * joint_features.window_size,
                    'end': (i - 1) * joint_features.window_size
                })
                start = i
                current_num = sliding_windows[i]

        # Adding the last group
        groups.append({
            'type': workout_types[current_num],
            'start': start * joint_features.window_size,
            'end': (len(sliding_windows) - 1) * joint_features.window_size
        })

        return groups


    def compute(self) -> None:
        joints_history = self._build_joints_history()
        sliding_windows = self.classifier.predict(joints_history)

        workouts = self._parse_sliding_windows(sliding_windows)
        workouts = self._update_times(workouts)
        workouts = self._filter_invalid(workouts)
        workouts = self._merge_consecutive(workouts)

        workouts_detected = [x['type'] for x in workouts]
        print(f"\n[bold]The following movements were classified in the video:[/bold] {' | '.join(workouts_detected)}\n")

        self._output = {
            "workouts": workouts,
            **self.input.get_carry()
        }


    def get_output(self) -> Tuple[ForwardStageAdvice, DataStageClassify]:
        return ForwardStageAdvice(), DataStageClassify(**self._output)