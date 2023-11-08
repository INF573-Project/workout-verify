from mmpose.apis import MMPoseInferencer

img_path = 'data/models/squat.jpg'

inferencer = MMPoseInferencer('human')
result_generator = inferencer(img_path, show=True)
result = next(result_generator)