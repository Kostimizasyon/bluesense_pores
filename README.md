# Bluesense Pores

The repo for preprocessing and segmentation for pores @ bluesense

This was the merging of two different projects and i couldnt be bothered to branch it out afterwards, maybe ill do it someday.
Also for some reason gitignore kept breaking so i had to reinit the project twice or so,
meaning we dont got too many commits & it was private for a lil while that too

Before running anything that uses `landmark_parse`, fetch the face landmarker
model (not committed to git due to size):
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Make sure to create a .env file with the structure:

bluesense_pore_detection id for the best segmentation model ive trained

```
WORKSPACE_NAME=
API_KEY=
MODEL_ID=
```
## Note for update:

Passing a `Dict[str, np.ndarray]` into `DetectionDataset` is deprecated in `0.30.0` and will be removed in `0.33.0`. Use a list of paths `List[str]` instead.

Should probably update / checkout api.py
