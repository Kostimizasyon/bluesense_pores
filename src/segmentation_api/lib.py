import os
import cv2
import supervision as sv
from roboflow import Roboflow
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# i just asked claude to implement a the last lambda part for a pos / neg answer and not only
# did it not do that it also deleted all of my comments and im too lazy to recomment rn

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


def annotate_path(path: str, confidence: float, class_name: str, export_type: int = 0, question_interval_percentage: int = 100):

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

        file_list = os.listdir(path)
        length = len(file_list)

        interval = max(1, round(length * question_interval_percentage / 100))

        for count, file_name in enumerate(file_list):

            file_path = os.path.join(path, file_name)

            if not os.path.isfile(file_path):
                continue

            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in VALID_EXTS:
                continue

            image = cv2.imread(file_path)
            if image is None:
                continue

            with CLIENT.use_configuration(inference_configuration=confidance_config):
                result = CLIENT.infer(inference_input=image, model_id=MODEL_ID)

            if not result:
                raise RuntimeError("Couldnt get image inference")

            image_detections = sv.Detections.from_inference(result)

            detections[file_name] = image_detections
            images[file_name] = image

            should_check = (count == 0) or ((count + 1) % interval == 0)

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

        dataset = sv.DetectionDataset(
            classes=[class_name],
            images=images,
            annotations=detections
        )

    else:
        file_name = os.path.basename(path)
        image = cv2.imread(path)

        with CLIENT.use_configuration(inference_configuration=confidance_config):
            result = CLIENT.infer(inference_input=image, model_id=MODEL_ID)

        if not result:
            raise RuntimeError("Couldnt get image inference")

        image_detections = sv.Detections.from_inference(result)

        dataset = sv.DetectionDataset(
            classes=[class_name],
            images={file_name: image},
            annotations={file_name: image_detections}
        )
    return dataset

def save_dataset(dataset, export_type : int = 0, img_dir : str | None = None, output_dir : str = ""):
    if not os.path.isdir(output_dir):
        raise ValueError("Output dir is not a valid output dir")
    if img_dir is not None and img_dir is not os.path.isdir(img_dir):
        raise ValueError("Img dir is not a valid output dir")
    
    match export_type:
        case 0:
            dataset.as_coco(
                images_directory_path=img_dir,
                annotations_path=os.path.join(output_dir, "_annotations.coco.json"),
            )
        case 1:
            dataset.as_createml(
                images_directory_path=img_dir,
                annotations_path=os.path.join(output_dir, "_annotations.createml.json"),
            )
        case 2:
            dataset.as_pascal_voc(
                images_directory_path=img_dir,
                annotations_directory_path=os.path.join(output_dir, "annotations"),
            )
        case 3:
            dataset.as_labelme(
                images_directory_path=img_dir,
                annotations_directory_path=os.path.join(output_dir, "annotations"),
            )
        case 4:
            dataset.as_yolo(
                images_directory_path=img_dir,
                annotations_directory_path=os.path.join(output_dir, "labels"),
                data_yaml_path=os.path.join(output_dir, "data.yaml"),
            )
        case _:
            raise ValueError("export_type must be 0-4")

    return output_dir

if __name__ == "__main__":
    img = segment_and_save(path="57_PNG.jpg", confidance=0.2, save_path="dir/")
    cv2.imshow("Image", img)
    cv2.waitKey(0)