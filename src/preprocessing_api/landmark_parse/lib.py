import os
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

# Give the path to the task model
MODEL_PATH = ""

# Standard drawing specs for custom/partial region overlays
DEFAULT_NOSE_SPEC = drawing_utils.DrawingSpec(
    color=(224, 224, 224), thickness=1
)
DEFAULT_CUSTOM_SPEC = drawing_utils.DrawingSpec(
    color=(255, 255, 255), thickness=1
)

NOSE_INDICES = {
    1, 2, 4, 5, 6, 19, 94, 97, 98, 115, 122, 168,
    195, 197, 236, 240, 274, 275, 351, 399, 419, 456,
}

LEFT_CHEEK_INDICES = {
    50, 101, 116, 117, 118, 119, 120, 121, 126, 142,
    147, 187, 203, 205, 206, 207, 213, 214,
}

RIGHT_CHEEK_INDICES = {
    280, 330, 345, 346, 347, 348, 349, 350, 355, 371,
    376, 411, 423, 425, 426, 427, 433, 434,
}

FOREHEAD_INDICES = {
    10, 108, 109, 151, 337, 9, 107, 336, 8,
}

BUTTERFLY_ZONE_INDICES = (
    NOSE_INDICES | LEFT_CHEEK_INDICES | RIGHT_CHEEK_INDICES
)

def _filtered_connections(index_set, mesh_style):
  source = (
      vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
      if mesh_style == "tesselation"
      else vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
  )
  return [c for c in source if c.start in index_set and c.end in index_set]

def resize_and_show(
    image_path: str = None,
    image: np.ndarray = None,
    image_height=748,
    image_width=640,
):
  if image_path is not None and image is not None:
    raise ValueError(
        "Provide either 'image_path' OR 'image' array, not both."
    )
  elif image_path is None and image is None:
    raise ValueError("Provide at least 'image_path' or 'image' array.")
  elif image_path is not None:
    image_brg = cv2.imread(image_path)
  else:
    # Convert RGB (MediaPipe) to BGR (OpenCV) for correct color rendering
    image_brg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  h, w = image_brg.shape[:2]
  if h < w:
    image_brg = cv2.resize(image, (image_width, math.floor(h / (w / image_height))))
  else:
    image_brg = cv2.resize(image, (math.floor(w / (h / image_height)), image_height))

  image_rgb = cv2.cvtColor(image_brg, cv2.COLOR_BGR2GRAY)

  cv2.imshow("Preview", image_rgb)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def _draw_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result,
    draw_tesselation: bool = False,
    draw_contours: bool = False,
    draw_butterfly_only: bool = False,
    draw_nose: bool = False,
    draw_left_cheek: bool = False,
    draw_right_cheek: bool = False,
    draw_forehead: bool = False,
    draw_eyes: bool = False,
    draw_eyebrows: bool = False,
    draw_lips: bool = False,
    draw_face_oval: bool = False,
    custom_connections: list = None,
):
  any_selected = (
      any([
          draw_tesselation,
          draw_contours,
          draw_butterfly_only,
          draw_nose,
          draw_left_cheek,
          draw_right_cheek,
          draw_forehead,
          draw_eyes,
          draw_eyebrows,
          draw_lips,
          draw_face_oval,
      ])
      or custom_connections is not None
  )

  if not any_selected:
    raise ValueError(
        "Nothing selected to draw. Set at least one draw_* flag to True "
        "(e.g. draw_butterfly_only=True) or pass custom_connections."
    )

  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  for face_landmarks in face_landmarks_list:

    if draw_tesselation:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
      )

    if draw_contours:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
      )

    if draw_butterfly_only:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=_filtered_connections(
              BUTTERFLY_ZONE_INDICES, "tesselation"
          ),
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
      )

    # Uses DEFAULT_NOSE_SPEC (DrawingSpec) instead of contour style dict
    if draw_nose:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE,
          landmark_drawing_spec=None,
          connection_drawing_spec=DEFAULT_NOSE_SPEC,
      )

    if draw_left_cheek:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=_filtered_connections(LEFT_CHEEK_INDICES, "tesselation"),
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
      )

    if draw_right_cheek:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=_filtered_connections(
              RIGHT_CHEEK_INDICES, "tesselation"
          ),
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
      )

    if draw_forehead:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=_filtered_connections(FOREHEAD_INDICES, "tesselation"),
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
      )

    if draw_eyes:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=(
              vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
              + vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
          ),
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
      )

    if draw_eyebrows:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=(
              vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
              + vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
          ),
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
      )

    if draw_lips:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
      )

    if draw_face_oval:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
      )

    # Uses DEFAULT_CUSTOM_SPEC (DrawingSpec) instead of contour style dict
    if custom_connections is not None:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=custom_connections,
          landmark_drawing_spec=None,
          connection_drawing_spec=DEFAULT_CUSTOM_SPEC,
      )

  return annotated_image

def _get_annotated_image_and_detection_result(
    image_path: str,
    model_path: str,
    draw_tesselation: bool = False,
    draw_contours: bool = False,
    draw_butterfly_only: bool = False,
    draw_nose: bool = False,
    draw_left_cheek: bool = False,
    draw_right_cheek: bool = False,
    draw_forehead: bool = False,
    draw_eyes: bool = False,
    draw_eyebrows: bool = False,
    draw_lips: bool = False,
    draw_face_oval: bool = False,
    custom_connections: list = None,
    
):
  base_options = python.BaseOptions(model_asset_path=model_path)
  options = vision.FaceLandmarkerOptions(
      base_options=base_options,
      output_face_blendshapes=True,
      output_facial_transformation_matrixes=True,
      num_faces=1,
  )
  detector = vision.FaceLandmarker.create_from_options(options)

  image = mp.Image.create_from_file(image_path)
  detection_result = detector.detect(image)

  if detection_result.face_landmarks == []:
    raise ValueError(f"No face found on image: {image_path}")

  annotated_image = _draw_landmarks_on_image(
      image.numpy_view(),
      detection_result,
      draw_tesselation=draw_tesselation,
      draw_contours=draw_contours,
      draw_butterfly_only=draw_butterfly_only,
      draw_nose=draw_nose,
      draw_left_cheek=draw_left_cheek,
      draw_right_cheek=draw_right_cheek,
      draw_forehead=draw_forehead,
      draw_eyes=draw_eyes,
      draw_eyebrows=draw_eyebrows,
      draw_lips=draw_lips,
      draw_face_oval=draw_face_oval,
      custom_connections=custom_connections,
  )

  return annotated_image, detection_result

def crop_to_landmarks(
    image_rgb: np.ndarray,
    detection_result,
    padding_ratio: float = 0,
    crop_nose: bool = False,
    crop_left_cheek: bool = False,
    crop_right_cheek: bool = False,
    crop_forehead: bool = False,
    crop_butterfly: bool = False,
):
  """Crops to the bounding box of the given landmarks.

  If no crop_* flag is set, crops to ALL landmarks (whole face) — same
  behavior as before. If any crop_* flag is set, restricts the bounding
  box to the union of the selected zone(s) instead.
  """
  if detection_result.face_landmarks == []:
    raise ValueError("No face found in detection_result.")

  h, w = image_rgb.shape[:2]
  landmarks = detection_result.face_landmarks[0]

  zone_flags = [crop_nose, crop_left_cheek, crop_right_cheek, crop_forehead, crop_butterfly]
  zone_sets = [NOSE_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES, FOREHEAD_INDICES, BUTTERFLY_ZONE_INDICES]

  if any(zone_flags):
    index_set = set().union(*[s for flag, s in zip(zone_flags, zone_sets) if flag])
    xs = [landmarks[i].x * w for i in index_set]
    ys = [landmarks[i].y * h for i in index_set]
  else:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

  x_min, x_max = min(xs), max(xs)
  y_min, y_max = min(ys), max(ys)

  box_w = x_max - x_min
  box_h = y_max - y_min
  pad_x = box_w * padding_ratio
  pad_y = box_h * padding_ratio

  x_min = max(0, int(x_min - pad_x))
  y_min = max(0, int(y_min - pad_y))
  x_max = min(w, int(x_max + pad_x))
  y_max = min(h, int(y_max + pad_y))

  return image_rgb[y_min:y_max, x_min:x_max]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def check_dir(
    dir_path: str,
    model_path : str,
    crop_nose: bool = False,
    crop_left_cheek: bool = False,
    crop_right_cheek: bool = False,
    crop_forehead: bool = False,
    crop_butterfly: bool = False,
    padding_ratio: float = 0.0,
    VALID_TYPES=VALID_EXTS,
    save_rgb=False,
    save_lab=False,
):
  """Scans a directory, crops each face to the requested zone(s) via
  crop_to_landmarks, and saves results under 'processed_output'.

  Output layout:
      - save_rgb=False (default): crops go flat into
        <dir_path>/processed_output/cropped_images/
      - save_rgb=True: one subfolder per image under
        <dir_path>/processed_output/<stem>/, containing the crop, the
        original RGB image, and (if save_lab=True) a LAB-converted copy.

  Args:
      dir_path: Directory containing input images.
      model_path: Path to the FaceLandmarker .task model file.
      crop_nose / crop_left_cheek / crop_right_cheek / crop_forehead /
          crop_butterfly: Which zone(s) to crop to. If none set, crops to
          the whole face (same default as crop_to_landmarks).
      padding_ratio: Passed through to crop_to_landmarks.
      VALID_TYPES: Allowed file extensions. Defaults to VALID_EXTS.
      save_rgb: If True, saves a per-image subfolder with the original
          RGB image alongside the crop, instead of a flat crop-only folder.
      save_lab: If True (and save_rgb is True), also saves a LAB-converted
          copy of the original image. Off by default — LAB belongs to the
          diagnosis stage, not core cropping.

  Note:
      Batch mode intentionally does NOT raise on a missing face — it
      prints a warning and skips the file, since one bad image shouldn't
      kill a run over a whole directory. Use crop_to_landmarks /
      _get_annotated_image_and_detection_result directly (via api.py's
      crop_image) if you want a hard failure on a single image instead.
  """
  out_root = os.path.join(dir_path, "processed_output")
  cropped_root = os.path.join(out_root, "cropped_images")

  if save_rgb:
    os.makedirs(out_root, exist_ok=True)
  else:
    os.makedirs(cropped_root, exist_ok=True)

  base_options = python.BaseOptions(model_asset_path=model_path)
  options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
  detector = vision.FaceLandmarker.create_from_options(options)

  for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)

    # Skip subdirectories
    if not os.path.isfile(file_path):
      continue

    # Ignore files with unsupported extensions
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in VALID_TYPES:
      continue

    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
      continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    image = mp.Image.create_from_file(file_path)
    detection_result = detector.detect(image)

    if not detection_result.face_landmarks:
      print(f"No face detected, skipping: {filename}")
      continue

    crop_image = crop_to_landmarks(
        img_rgb,
        detection_result,
        padding_ratio=padding_ratio,
        crop_nose=crop_nose,
        crop_left_cheek=crop_left_cheek,
        crop_right_cheek=crop_right_cheek,
        crop_forehead=crop_forehead,
        crop_butterfly=crop_butterfly,
    )
    crop_bgr = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)

    if save_rgb:
      img_dir = os.path.join(out_root, stem)
      os.makedirs(img_dir, exist_ok=True)
      cv2.imwrite(os.path.join(img_dir, f"{stem}_cropped.png"), crop_bgr)
      cv2.imwrite(os.path.join(img_dir, f"{stem}_rgb.png"), img_bgr)
      if save_lab:
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        cv2.imwrite(os.path.join(img_dir, f"{stem}_lab.png"), img_lab)
    else:
      cv2.imwrite(os.path.join(cropped_root, f"{stem}_cropped.png"), crop_bgr)

if __name__ == "__main__":
  IMAGE_PATH = "image.png"

  annotated_image, detection_result = _get_annotated_image_and_detection_result(
      image_path=IMAGE_PATH,
      model_path=MODEL_PATH,
      draw_butterfly_only=True
  )

  landmark_crop = crop_to_landmarks(
      annotated_image, detection_result=detection_result, padding_ratio=0.0
  )

  resize_and_show(image=landmark_crop)