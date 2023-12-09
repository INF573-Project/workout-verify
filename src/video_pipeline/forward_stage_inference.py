# External imports
from mmpose.apis import MMPoseInferencer
from pypipeline.stages import IForwardStage
from typing import Tuple
import numpy as np
import threading
import multiprocessing
from pathlib import Path
import cv2
import pickle

# Local imports
from .forward_stage_joints import ForwardStageJoints
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

    def process_slice(self, slice_data, inferencer):
        kpts = []

        for frame in slice_data:
            result_generator = inferencer(frame)
            
            for result in result_generator:
                kpts.append(np.array(result['predictions'][0][0]['keypoints']))

        return kpts

    def threaded_processing(self, data, num_threads, inferencer):
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
            
            inferencer = MMPoseInferencer(pose3d='human3d')
            result_generator = inferencer(self.input.video_path)
            
            kpts = []

            for result in result_generator:
                    kpts.append(result['predictions'][0][0]['keypoints'])
            
            with open('kpts.pickle', 'wb') as f:
                pickle.dump(kpts, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open('kpts.pickle', 'rb') as f:
                kpts = pickle.load(f)
        
        kpts = [np.array(x) for x in kpts]
        self._output = {"keypoints": kpts, **self.input.get_carry()}

    def get_output(self) -> Tuple[ForwardStageJoints, DataStageInference]:
        return ForwardStageJoints(), DataStageInference(**self._output)