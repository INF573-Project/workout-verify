# External imports
import json_tricks as json
import mmcv
import mimetypes
import os
import mmengine
import numpy as np
from mmengine.logging import print_log
from mmpose.apis import MMPoseInferencer
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector
from pypipeline.stages import IForwardStage
from typing import Tuple
import threading
import multiprocessing
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

    def process_one_image(img,
                      detector,
                      pose_estimator,
                      visualizer=None):
        # predict bbox
        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
        bboxes = bboxes[nms(bboxes, 0.3), :4]
        # predict keypoints
        pose_results = inference_topdown(pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)
        
        # show the results - visualizer stuff
        if isinstance(img, str):
            img = mmcv.imread(img, channel_order='rgb')
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)
        visualizer.add_datasample('result',
                              img,
                              data_sample=data_samples,
                              draw_gt=False,
                              draw_heatmap=False,
                              draw_bbox='store_true',
                              show_kpt_idx=True,
                              skeleton_style='mmpose',
                              show=False,
                              wait_time=0.001,
                              kpt_thr=0.1)
        # if there is no instance detected, return None
        return data_samples.get('pred_instances', None)


    def process_hand(video_path):
        # arguments for configs and video
        args = {}
        args['det_config'] = 'mmpose/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py'
        args['det_checkpoint'] = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth'
        args['pose_config'] = 'mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py'
        args['pose_checkpoint'] = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
        args['output_root'] = 'vis_results' # Change output root to whatever here
        args['device'] = 'cpu'# 'cuda:0' # if you have cuda

        # prepare output file
        output_file = None
        mmengine.mkdir_or_exist(args['output_root'])
        output_file = os.path.join(args['output_root'],
                                        os.path.basename(video_path))

        args['pred_save_path'] = f"{args['output_root']}/results_{os.path.splitext(os.path.basename(video_path))[0]}.json"

        # build detector
        detector = init_detector(args['det_config'], args['det_checkpoint'], device=args['device'])
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

        # build pose estimator
        pose_estimator = init_pose_estimator(
            args['pose_config'],
            args['pose_checkpoint'],
            device=args['device'],
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        # build visualizer
        pose_estimator.cfg.visualizer.radius = 3
        pose_estimator.cfg.visualizer.alpha = 0.8
        pose_estimator.cfg.visualizer.line_width = 1
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style='mmpose')

        input_type = mimetypes.guess_type(video_path)[0].split('/')[0]

        # intialize video and video writer for output
        cap = cv2.VideoCapture(video_path)
        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            # read a frame
            success, frame = cap.read()
            frame_idx += 1

            if not success: # end of video
                break

            # topdown pose estimation
            pred_instances = process_one_image(frame, detector,
                                            pose_estimator, visualizer)
            # count fingers up from keypoints
            kpts = [np.array(pred_instances.keypoints[i]) for i in range(len(pred_instances.keypoints))]

            # save predictions
            pred_instances_list.append(
                dict(frame_id=frame_idx,
                    instances=split_instances(pred_instances)))

            # output videos
            frame_vis = visualizer.get_image() # get frame
            if video_writer is None: # first frame: initiate video_writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file,fourcc,25,(frame_vis.shape[1], frame_vis.shape[0]))

            video_writer.write(mmcv.rgb2bgr(frame_vis)) # write frame

        video_writer.release()
        cap.release()

        with open(args['pred_save_path'], 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list), f, indent='\t')
            print(f"predictions have been saved at {args['pred_save_path']}")

        return kpts
    
    def compute(self) -> None:
        cached = True

        if not cached:
            output_skeleton = self.input.video_skeleton_path
            Path(output_skeleton).mkdir(parents=True, exist_ok=True)
            
            inferencer_joints = MMPoseInferencer(pose3d='human3d')
            results_joints = inferencer_joints(self.input.video_path)
            results_hand = process_hand(self.input.video_path)
            
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
        self._output = {"keypoints_hands": kpts_hands,
                        "keypoints_joints": kpts_joints,
                        **self.input.get_carry()}

    def get_output(self) -> Tuple[ForwardStageJoints, ForwardStagehands, DataStageInference]:
        return ForwardStageJoints(), ForwardStagehands(), DataStageInference(**self._output)