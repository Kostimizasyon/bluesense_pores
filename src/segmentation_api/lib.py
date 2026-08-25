import os
import shutil
import time
import cv2
import supervision as sv
from roboflow import Roboflow
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# i just asked claude to implement a the last lambda part for a pos / neg answer and not only
# did it not do that it also deleted all of my comments and im too lazy to recomment rn

# Then i asked it to do a progress displayer as i was too lazy, now the code is kinda too long and im too lazy once again

load_dotenv()

# Valid file types
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# get .env
API_KEY = os.getenv("API_KEY")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
MODEL_ID = os.getenv("MODEL_ID")

def ask_input(prompt: str, check_lambda):
    while True:
        ans = input(prompt)
        if check_lambda(ans) == True:
            return ans
        else:
            continue

def segment_image(path: str, confidance: float):

    if confidance < 0 or confidance > 1:
        raise ValueError("Confidance should be a positive value between 0 and 1")

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

    confidance_config = InferenceConfiguration(confidence_threshold=confidance)

    image = cv2.imread(path)
    if image is None:
        raise ValueError("Couldn't read image")

    with CLIENT.use_configuration(inference_configuration=confidance_config):
        result = CLIENT.infer(inference_input=image, model_id=MODEL_ID)

    if not result:
        raise RuntimeError("Couldnt get image inference")

    detections = sv.Detections.from_inference(result)

    annotator = sv.MaskAnnotator()
    annotated = annotator.annotate(scene=image.copy(), detections=detections)
    return annotated


def segment_and_save(path: str, confidance: float, save_path: str = None):

    image = segment_image(path, confidance)
    file_name = os.path.basename(path)

    if save_path is not None:
        out_path = f"{save_path}SEGMENTED_{file_name}"
    else:
        out_path = f"SEGMENTED_{file_name}"

    cv2.imwrite(out_path, image)
    return image

def annotate_path(path: str, confidence: float, class_name: str,
                   question_interval_percentage: int = 100,
                   progress_interval_percentage: int = 10, verbose: bool = False):
    if not (os.path.isdir(path) or os.path.splitext(path)[1].lower() in VALID_EXTS):
        raise ValueError("Path should be an image file or a directory")

    if confidence < 0 or confidence > 1:
        raise ValueError("Confidance should be a positive value between 0 and 1")

    if class_name.strip() == "":
        raise ValueError("Give a proper class name")

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

    confidance_config = InferenceConfiguration(confidence_threshold=confidence)

    detections = {}
    images = {}

    if os.path.isdir(path):
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()

        file_list = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
        total = len(file_list)

        if total == 0:
            raise ValueError(f"No valid images found in directory: {path}")

        question_interval = max(1, round(total * question_interval_percentage / 100))
        progress_interval = max(1, round(total * progress_interval_percentage / 100))

        succeeded = 0
        failed = 0
        skipped_unreadable = 0

        print(f"Found {total} images to annotate.")

        for count, file_name in enumerate(file_list):

            file_path = os.path.join(path, file_name)

            image = cv2.imread(file_path)
            if image is None:
                skipped_unreadable += 1
                if verbose:
                    print(f"Couldn't read, skipping: {file_name}")
                continue

            # A single image's inference failing (network hiccup, rate
            # limit, corrupt file the API rejects, etc.) shouldn't kill the
            # whole batch. Catch, count, skip, keep going.
            try:
                with CLIENT.use_configuration(inference_configuration=confidance_config):
                    result = CLIENT.infer(inference_input=image, model_id=MODEL_ID)

                if not result:
                    raise RuntimeError("Inference returned no result")

                image_detections = sv.Detections.from_inference(result)

            except Exception as e:
                failed += 1
                if verbose:
                    print(f"Inference failed for {file_name}: {e}")
                continue

            detections[file_name] = image_detections
            images[file_name] = image
            succeeded += 1

            if verbose:
                print(f"Annotated: {file_name}")

            processed = count + 1

            # Progress notification — independent of the interactive
            # confidence check below, always fires regardless of verbose.
            if not verbose and (processed % progress_interval == 0 or processed == total):
                pct = round(processed / total * 100)
                print(f"  ...{processed}/{total} processed ({pct}%) — "
                      f"{succeeded} succeeded, {failed} failed, {skipped_unreadable} unreadable")

            # Interactive confidence check-in, only if the user opted in.
            if question_interval_percentage < 100:
                should_check = count % question_interval == 0

                if should_check:
                    annotated = mask_annotator.annotate(scene=image.copy(), detections=image_detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=image_detections)

                    cv2.imshow("Original", image)
                    cv2.imshow("Segmented", annotated)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    yn_check = lambda x: x.strip().upper() in ("Y", "N")
                    ans = ask_input("Change confidence? Y / N: ", yn_check)

                    if ans.strip().upper() == "Y":
                        def is_valid_conf(x):
                            try:
                                v = float(x)
                                return 0 <= v <= 1
                            except ValueError:
                                return False

                        new_conf = ask_input(f"Current confidance = {confidence} \n Enter new confidence (0-1): ", is_valid_conf)
                        confidence = float(new_conf)
                        confidance_config = InferenceConfiguration(confidence_threshold=confidence)

        print("\nAnnotation complete.")
        print(f"Succeeded: {succeeded}")
        print(f"Failed:    {failed}")
        print(f"Unreadable: {skipped_unreadable}")

        if succeeded == 0:
            raise RuntimeError("No images were successfully annotated — check the failures above before saving a dataset.")

        dataset = sv.DetectionDataset(
            classes=[class_name],
            images=images,
            annotations=detections
        )

    else:
        file_name = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Couldn't read image: {path}")

        try:
            with CLIENT.use_configuration(inference_configuration=confidance_config):
                result = CLIENT.infer(inference_input=image, model_id=MODEL_ID)

            if not result:
                raise RuntimeError("Inference returned no result")

            image_detections = sv.Detections.from_inference(result)

        except Exception as e:
            raise RuntimeError(f"Inference failed for {file_name}: {e}") from e

        dataset = sv.DetectionDataset(
            classes=[class_name],
            images={file_name: image},
            annotations={file_name: image_detections}
        )
    return dataset

def save_dataset(dataset, img_dir: str, output_dir: str, export_type: int = 0,
                 verbose: bool = False, copy_images: bool = False):
    if not os.path.isdir(img_dir):
        raise ValueError("Img dir is not a valid dir")

    images_dir = img_dir
    if copy_images:
        images_dir = os.path.join(output_dir, "deduped")
        if os.path.isdir(images_dir):
            shutil.rmtree(images_dir)
        os.makedirs(images_dir)

        for file_name in os.listdir(img_dir):
            source_path = os.path.join(img_dir, file_name)
            if (
                os.path.isfile(source_path)
                and os.path.splitext(file_name)[1].lower() in VALID_EXTS
            ):
                shutil.copy2(source_path, os.path.join(images_dir, file_name))
    else:
        os.makedirs(output_dir, exist_ok=True)

    annotation_dir = images_dir if copy_images else output_dir

    print(f"Writing annotations (export_type={export_type})...")
    match export_type:
        case 0:
            dataset.as_coco(
                images_directory_path=images_dir,
                annotations_path=os.path.join(annotation_dir, "_annotations.coco.json"),
            )
        case 1:
            dataset.as_createml(
                images_directory_path=images_dir,
                annotations_path=os.path.join(annotation_dir, "_annotations.createml.json"),
            )
        case 2:
            dataset.as_pascal_voc(
                images_directory_path=images_dir,
                annotations_directory_path=os.path.join(annotation_dir, "annotations"),
            )
        case 3:
            dataset.as_labelme(
                images_directory_path=images_dir,
                annotations_directory_path=os.path.join(annotation_dir, "annotations"),
            )
        case 4:
            dataset.as_yolo(
                images_directory_path=images_dir,
                annotations_directory_path=os.path.join(annotation_dir, "labels"),
                data_yaml_path=os.path.join(annotation_dir, "data.yaml"),
            )
        case _:
            raise ValueError("export_type must be 0-4")

    print(f"Dataset saved to: {annotation_dir}")
    return annotation_dir

def splice_dir(dir : str,interval : int = 25):

    if not os.path.isdir(dir):
        raise ValueError("Provided path is not a directory")

    if interval <= 0 or interval > 100 or interval % 5 != 0:
        raise ValueError("Interval must be a positive integer between 1 and 100 dividable by 5")

    files = [
        file_name for file_name in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, file_name))
        and os.path.splitext(file_name)[1].lower() in VALID_EXTS
    ]
    total = len(files)
    if total == 0:
        raise ValueError("No files found in the directory")

    split_count = 100 // interval
    split_size = total // split_count
    splice_paths = []

    for split_number in range(split_count):
        start = split_number * split_size
        end = total if split_number == split_count - 1 else start + split_size
        subset_dir = os.path.join(dir, f"{interval}percent_{split_number + 1}")
        if os.path.isdir(subset_dir):
            shutil.rmtree(subset_dir)
        os.makedirs(subset_dir)

        splice_paths.append(subset_dir)

        for file_name in files[start:end]:
            source_path = os.path.join(dir, file_name)
            dest_path = os.path.join(subset_dir, file_name)
            shutil.copy2(source_path, dest_path)

        print(f"Copied {end - start} files to {subset_dir}")

    return splice_paths

if __name__ == "__main__":
    IMG_PATH = r""
    PROJECT_ID = ""
    CLASS_NAME = ""
    validate   = False

    split_paths = splice_dir(IMG_PATH, interval=25)
    
    for path in split_paths:
        dataset = annotate_path(path, confidence=0.15, class_name=CLASS_NAME,
                                question_interval_percentage=100,
                                progress_interval_percentage=10, verbose=False)
        save_dataset(dataset, img_dir=path, output_dir=path, export_type=0, verbose=False, copy_images=False)
        print(f"Annotated and saved dataset for {path}")
        if validate:
            input("Enter to continue to the next batch...")

    print(f"Spliced directories: {split_paths}")