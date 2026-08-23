"""
api.py — Unified pipeline interface for the bluesense_pores_pp preprocessing
modules.

Wires dupe_detect, landmark_parse (crop), rgb2lab, and the segmentation
module's dataset save utility into a single importable API + CLI.
"""
import os
import cv2
import src.preprocessing_api.rgb2lab.lib as r2l
import src.preprocessing_api.dupe_detect.lib as dd
import src.preprocessing_api.landmark_parse.lib as lp
import src.segmentation_api.lib as seg

# I made all the lil libs and told claude to just do a api.py file so its all its fault if anything is wrong
# Im totally in the right

# TODO Give the path to the task model
MODEL_PATH = ""

VALID_ZONES = {"nose", "left_cheek", "right_cheek", "forehead", "butterfly", "full_face"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 1. Dedupe
# ---------------------------------------------------------------------------

def dedupe_dir(dir_path: str, hash_size: int = 8, threshold: int = 5,
               delete: bool = False, report: bool = True) -> dict:
    """Finds near-duplicate images in a directory via perceptual hashing.

    Args:
        dir_path: Directory to scan.
        hash_size: pHash grid size — see dupe_detect/lib.py for tuning notes.
        threshold: Max Hamming distance to count as a duplicate.
        delete: If True, deletes every image in a cluster except the first
            (alphabetically). Destructive — off by default.
        report: If True, prints dupe_detect's summary report.

    Returns:
        dict with "clusters" (raw cluster map), "kept" (one filename per
        cluster), and "removed" (filenames deleted or flagged for removal).
    """
    clusters = dd.find_duplicates(dir_path, hash_size=hash_size, threshold=threshold)

    if report:
        dd.print_report(clusters)

    kept, removed = [], []
    for files in clusters.values():
        kept.append(files[0])
        removed.extend(files[1:])

    if delete:
        for f in removed:
            os.remove(os.path.join(dir_path, f))

    return {"clusters": clusters, "kept": kept, "removed": removed}


# ---------------------------------------------------------------------------
# 2. Crop to facial zone
# ---------------------------------------------------------------------------

def _zone_kwargs(zone: str) -> dict:
    if zone not in VALID_ZONES:
        raise ValueError(f"zone must be one of {sorted(VALID_ZONES)}, got {zone!r}")

    return {
        "crop_nose": zone == "nose",
        "crop_left_cheek": zone == "left_cheek",
        "crop_right_cheek": zone == "right_cheek",
        "crop_forehead": zone == "forehead",
        "crop_butterfly": zone == "butterfly",
        # full_face: every flag stays False -> crop_to_landmarks uses all landmarks
    }


def crop_image(image_path: str, zone: str = "butterfly", model_path: str = MODEL_PATH,
               padding_ratio: float = 0.0):
    """Crops a single image to the given facial zone.

    Unlike crop_dir/check_dir (which is built for batch runs and skips
    faceless images with a printed warning), this raises immediately if no
    face is detected — use this when a missing face should be a hard
    failure rather than something silently skipped.

    Args:
        image_path: Path to a single image file.
        zone: One of VALID_ZONES.
        model_path: Path to the FaceLandmarker .task file.
        padding_ratio: Extra padding around the zone's bounding box.

    Returns:
        np.ndarray: the cropped image, in RGB.

    Raises:
        ValueError: if zone is invalid, or if no face is detected in the
            image (propagated from landmark_parse.lib).
    """
    zone_kwargs = _zone_kwargs(zone)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # _get_annotated_image_and_detection_result raises ValueError itself
    # if detection_result.face_landmarks == [] (no face found on image).
    _, detection_result = lp._get_annotated_image_and_detection_result(
        image_path=image_path,
        model_path=model_path,
        draw_butterfly_only=True,  # any draw_* flag works, we only need detection_result
    )

    return lp.crop_to_landmarks(
        img_rgb,
        detection_result,
        padding_ratio=padding_ratio,
        **zone_kwargs,
    )


def crop_dir(dir_path: str, zone: str = "butterfly", model_path: str = MODEL_PATH,
             padding_ratio: float = 0.0, save_rgb: bool = False, save_lab: bool = False) -> str:
    """Crops every detected face in dir_path to the given zone.

    Batch version — faceless images are skipped with a printed warning
    rather than raising (see crop_image for the hard-failure single-image
    version).

    Args:
        dir_path: Directory of input images.
        zone: One of VALID_ZONES. "full_face" crops to all landmarks
            (landmark_parse's default when no crop_* flag is set).
        model_path: Path to the FaceLandmarker .task file.
        padding_ratio: Extra padding around the zone's bounding box.
        save_rgb: Keep a copy of the original next to each crop.
        save_lab: Also save a LAB copy (only applies if save_rgb=True).

    Returns:
        Path to the directory containing the crops.
    """
    zone_kwargs = _zone_kwargs(zone)

    lp.check_dir(
        dir_path=dir_path,
        model_path=model_path,
        padding_ratio=padding_ratio,
        save_rgb=save_rgb,
        save_lab=save_lab,
        **zone_kwargs,
    )

    out_root = os.path.join(dir_path, "processed_output")
    return out_root if save_rgb else os.path.join(out_root, "cropped_images")


# ---------------------------------------------------------------------------
# 3. LAB conversion
# ---------------------------------------------------------------------------

def lab_convert_dir(dir_path: str, depth_scanner: bool = False, save_rgb: bool = False) -> str:
    """Converts every image in dir_path to CIELAB, optionally with per-channel
    heatmaps (L/A/B/gradient) for visual inspection.

    Args:
        dir_path: Directory of input images (e.g. crop_dir's output).
        depth_scanner: If True, also saves L/A/B/Sobel-gradient heatmaps.
        save_rgb: If True, keeps a restored RGB copy alongside the LAB output.

    Returns:
        Path to the 'processed_output' directory.
    """
    r2l.check_dir(dir_path, depth_scanner=depth_scanner, save_rgb=save_rgb)
    return os.path.join(dir_path, "processed_output")


# ---------------------------------------------------------------------------
# 4. Dataset save (segmentation.lib passthrough)
# ---------------------------------------------------------------------------

def save_dataset(dataset, export_type: int = 0, img_dir: str | None = None, dataset_dir: str = "") -> str:
    """Writes an sv.DetectionDataset to disk in the requested annotation
    format. Thin wrapper around segmentation.lib.save_dataset.

    Args:
        dataset: An sv.DetectionDataset (e.g. from segmentation.lib.annotate_path).
        export_type: 0=coco, 1=createml, 2=pascal_voc, 3=labelme, 4=yolo.
        img_dir: Directory to write images into. Defaults to
            <dataset_dir>/images if not given.
        dataset_dir: Directory to write annotations (and default images) into.
            Must already exist.

    Returns:
        The dataset_dir the dataset was written into.
    """
    return seg.save_dataset(dataset, export_type=export_type, img_dir=img_dir, dataset_dir=dataset_dir)


# ---------------------------------------------------------------------------
# 5. Full pipeline: dedupe -> crop -> LAB -> segment -> save
# ---------------------------------------------------------------------------

def run_pipeline(dir_path: str, zone: str = "butterfly", model_path: str = MODEL_PATH,
                  dedupe: bool = True, delete_dupes: bool = False,
                  padding_ratio: float = 0.0, depth_scanner: bool = False) -> dict:
    """Runs preprocessing only: dedupe -> crop to zone -> LAB-convert the crops.

    Args:
        dir_path: Directory of raw input images.
        zone: Facial zone to crop to (see VALID_ZONES).
        model_path: Path to the FaceLandmarker .task file.
        dedupe: If True, runs dedupe_dir first.
        delete_dupes: If True, deletes duplicate files (destructive).
        padding_ratio: Passed through to crop_dir.
        depth_scanner: Passed through to lab_convert_dir.

    Returns:
        dict with "dedupe_result" (or None), "cropped_dir", "lab_dir".
    """
    dedupe_result = dedupe_dir(dir_path, delete=delete_dupes) if dedupe else None
    cropped_dir = crop_dir(dir_path, zone=zone, model_path=model_path, padding_ratio=padding_ratio)
    lab_dir = lab_convert_dir(cropped_dir, depth_scanner=depth_scanner)

    return {"dedupe_result": dedupe_result, "cropped_dir": cropped_dir, "lab_dir": lab_dir}


def full_pipeline(dir_path: str, zone: str = "butterfly", model_path: str = MODEL_PATH,
                   dedupe: bool = True, delete_dupes: bool = False, padding_ratio: float = 0.0,
                   depth_scanner: bool = False, confidence: float = 0.1, class_name: str = "pore",
                   export_type: int = 4, question_interval_percentage: int = 100,
                   dataset_dir: str | None = None) -> dict:
    """Runs the full pipeline end to end: preprocessing (dedupe -> crop ->
    LAB) followed by Roboflow-bootstrap segmentation on the cropped images,
    then saving the resulting dataset to disk.

    Segmentation runs on cropped_dir (RGB crops), not lab_dir — the
    Roboflow model expects ordinary RGB input; LAB output is a separate
    diagnostic artifact, not segmentation input.

    Args:
        dir_path, zone, model_path, dedupe, delete_dupes, padding_ratio,
            depth_scanner: same as run_pipeline.
        confidence: Confidence threshold for Roboflow inference.
        class_name: Class name to assign detections (e.g. "pore").
        export_type: Export format for the resulting dataset
            (0=coco, 1=createml, 2=pascal_voc, 3=labelme, 4=yolo).
        question_interval_percentage: How often (as % of images processed)
            the segmentation step pauses to ask whether to change confidence.
        dataset_dir: Directory to save the exported dataset into. Defaults
            to <cropped_dir>/dataset if not given.

    Returns:
        dict with "dedupe_result", "cropped_dir", "lab_dir", "dataset_dir".
    """
    preprocess_result = run_pipeline(
        dir_path=dir_path, zone=zone, model_path=model_path, dedupe=dedupe,
        delete_dupes=delete_dupes, padding_ratio=padding_ratio, depth_scanner=depth_scanner,
    )
    cropped_dir = preprocess_result["cropped_dir"]

    dataset = seg.annotate_path(
        cropped_dir, confidence, class_name,
        export_type=export_type, question_interval_percentage=question_interval_percentage,
    )

    if dataset_dir is None:
        dataset_dir = os.path.join(cropped_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    save_dataset(dataset, export_type=export_type, dataset_dir=dataset_dir)

    return {**preprocess_result, "dataset_dir": dataset_dir}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bluesense pore pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preprocess: dedupe -> crop -> LAB, no segmentation ---
    pre = subparsers.add_parser("preprocess", help="Dedupe -> crop -> LAB convert only")
    pre.add_argument("--dir_path", type=str, required=True)
    pre.add_argument("--zone", type=str, default="butterfly", choices=sorted(VALID_ZONES))
    pre.add_argument("--model_path", type=str, default=MODEL_PATH)
    pre.add_argument("--no_dedupe", action="store_true")
    pre.add_argument("--delete_dupes", action="store_true")
    pre.add_argument("--padding_ratio", type=float, default=0.0)
    pre.add_argument("--depth_scanner", action="store_true")

    # --- full: preprocess -> segment -> save ---
    full_p = subparsers.add_parser("full", help="Preprocess + segment + save end to end")
    full_p.add_argument("--dir_path", type=str, required=True)
    full_p.add_argument("--zone", type=str, default="butterfly", choices=sorted(VALID_ZONES))
    full_p.add_argument("--model_path", type=str, default=MODEL_PATH)
    full_p.add_argument("--no_dedupe", action="store_true")
    full_p.add_argument("--delete_dupes", action="store_true")
    full_p.add_argument("--padding_ratio", type=float, default=0.0)
    full_p.add_argument("--depth_scanner", action="store_true")
    full_p.add_argument("--confidence", type=float, default=0.1)
    full_p.add_argument("--class_name", type=str, default="pore")
    full_p.add_argument("--export_type", type=int, default=4, choices=range(5))
    full_p.add_argument("--question_interval_percentage", type=int, default=100)
    full_p.add_argument("--dataset_dir", type=str, default=None)

    args = parser.parse_args()

    if args.command == "preprocess":
        result = run_pipeline(
            dir_path=args.dir_path, zone=args.zone, model_path=args.model_path,
            dedupe=not args.no_dedupe, delete_dupes=args.delete_dupes,
            padding_ratio=args.padding_ratio, depth_scanner=args.depth_scanner,
        )
        print(f"\nCropped images: {result['cropped_dir']}")
        print(f"LAB output:     {result['lab_dir']}")

    elif args.command == "full":
        result = full_pipeline(
            dir_path=args.dir_path, zone=args.zone, model_path=args.model_path,
            dedupe=not args.no_dedupe, delete_dupes=args.delete_dupes,
            padding_ratio=args.padding_ratio, depth_scanner=args.depth_scanner,
            confidence=args.confidence, class_name=args.class_name, export_type=args.export_type,
            question_interval_percentage=args.question_interval_percentage,
            dataset_dir=args.dataset_dir,
        )
        print(f"\nCropped images: {result['cropped_dir']}")
        print(f"LAB output:     {result['lab_dir']}")
        print(f"Dataset saved:  {result['dataset_dir']}")