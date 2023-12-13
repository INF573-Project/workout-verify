# External imports
from mmpose.apis import MMPoseInferencer
from pypipeline.stages import IForwardStage
from typing import Tuple
import numpy as np
import threading
from pathlib import Path
import cv2
import pickle

# Local imports
from .forward_stage_joints import ForwardStageJoints
from .forward_stage_hands import ForwardStagehands
from .schemas.data_stage_load import DataStageOutputLoad
from .schemas.data_stage_inference import DataStageInference


class ForwardStageInference(IForwardStage[DataStageOutputLoad, DataStageInference, ForwardStageJoints]):

    def load_video_as_np_array(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def process_slice(self, slice_data, inferencer_joints, inferencer_hands):
        kpts_joints = []

        for frame in slice_data:
            results_joints = inferencer_joints(frame)
            results_hands = inferencer_hands(frame)
            
            for result in results_joints:
                kpts_joints.append(np.array(result['predictions'][0][0]['keypoints']))
            kpts_hand = [np.array(results_hands.keypoints[i]) for i in range(len(results_hands.keypoints))]

        return kpts_joints, kpts_hand

    def threaded_processing(self, data, num_threads, inferencer): # TODO for hands
        data_slices = np.array_split(data, num_threads)

        results = [None] * num_threads

        def process_slice_thread(slice_idx):
            nonlocal results
            results[slice_idx] = self.process_slice(data_slices[slice_idx], inferencer)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=process_slice_thread, args=(i,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        final_result = np.concatenate(results)

        return final_result

    def compute(self) -> None:
        cached = True

        if not cached:
            output_skeleton = self.input.video_skeleton_path
            Path(output_skeleton).mkdir(parents=True, exist_ok=True)
            
            inferencer_joints = MMPoseInferencer(pose3d='human3d')
            results_joints = inferencer_joints(self.input.video_path)
            
            kpts_joints = []
            kpts_hands = []

            for result in results_joints:
                    kpts_joints.append(result['predictions'][0][0]['keypoints'])
            
            with open('kpts_joints.pickle', 'wb') as f:
                pickle.dump(kpts_joints, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('kpts_hands.pickle', 'wb') as f:
                pickle.dump(kpts_hands, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open('kpts_joints.pickle', 'rb') as f:
                kpts_joints = pickle.load(f)
            with open('kpts_hands.pickle', 'rb') as f:
                kpts_hands = pickle.load(f)
        
        kpts_joints = [np.array(x) for x in kpts_joints]
        kpts_hands = [np.array(x) for x in kpts_hands]
        
        self._output = {
            "keypoints_hands": kpts_hands,
            "keypoints": kpts_joints,
            **self.input.get_carry()
        }

    def get_output(self) -> Tuple[ForwardStageJoints, ForwardStagehands, DataStageInference]:
        return ForwardStagehands(), DataStageInference(**self._output)