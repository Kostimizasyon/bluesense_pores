# Bluesense Pores

The repo for preprocessing and segmentation for pores @ bluesense

Before running anything that uses `landmark_parse`, fetch the face landmarker
model (not committed to git due to size):
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Make sure to create a .env file with the structure:
WORKSPACE_NAME=
API_KEY=
MODEL_ID=