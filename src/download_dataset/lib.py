import os
import sys
from attrs import validate
import requests
from dotenv import load_dotenv
from roboflow import Roboflow

# Ensure the src directory is importable regardless of the current working directory.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing_api.dupe_detect.lib import find_duplicates, print_report

load_dotenv()

API_KEY = os.getenv("API_KEY")

rf = Roboflow(API_KEY)

def download_from_project(project, folder_name: str | None = None,
                           run_dedupe: bool = True, delete_dupes: bool = True,
                           verbose: bool = False, progress_interval_percentage: int = 10, validate: bool = False):

    if folder_name is None:
        folder_name = project.name
        if not folder_name:
            raise ValueError(
                "Couldn't get project name. Explicitly provide a folder name."
            )

    output_dir = os.path.join("raw_images", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        records = []
        for page in project.search_all(
            offset=0,
            limit=100,
            in_dataset=True,
            batch=False,
            fields=["id", "name", "owner"],
        ):
            records.extend(page)

    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve images from Roboflow: {e}"
        ) from e

    if not records:
        raise RuntimeError("Roboflow returned no images.")

    total = len(records)
    print(f"Found {total} images.")
    print(f"Downloading to: {output_dir}")

    interval = max(1, round(total * progress_interval_percentage / 100))

    downloaded = 0
    failed = 0

    for count, record in enumerate(records):
        try:
            filename = record["name"]
            owner = record["owner"]
            image_id = record["id"]

        except KeyError as e:
            failed += 1
            if verbose:
                print(f"Invalid Roboflow record, missing field: {e}")
            continue

        # Always suffix with image_id — guaranteed unique per Roboflow
        # record, unlike the original filename. Prevents two different
        # images silently overwriting each other mid-download if their
        # source names collide.
        stem, ext = os.path.splitext(filename)
        if not ext:
            ext = ".jpg"  # source.roboflow.com always serves original.jpg regardless
        unique_filename = f"{stem}_{image_id}{ext}"
        path = os.path.join(output_dir, unique_filename)

        url = (
            f"https://source.roboflow.com/"
            f"{owner}/{image_id}/original.jpg"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(path, "wb") as f:
                f.write(response.content)

            downloaded += 1
            if verbose:
                print(f"Downloaded: {unique_filename}")

        except requests.RequestException as e:
            failed += 1
            if verbose:
                print(f"Failed to download {unique_filename}: {e}")

        except OSError as e:
            failed += 1
            if verbose:
                print(f"Failed to save {unique_filename}: {e}")

        processed = count + 1
        if not verbose and (processed % interval == 0 or processed == total):
            pct = round(processed / total * 100)
            print(f"  ...{processed}/{total} processed ({pct}%) — "
                  f"{downloaded} downloaded, {failed} failed")

    print("\nDownload complete.")
    print(f"Downloaded: {downloaded}")
    print(f"Failed:     {failed}")

    if run_dedupe:
        print("\nRunning duplicate check on downloaded batch...")
        clusters = find_duplicates(output_dir)
        print_report(clusters)

        if delete_dupes and len(clusters) > 0:
            removed = 0
            for files in clusters.values():
                for f in files[1:]:
                    os.remove(os.path.join(output_dir, f))
                    removed += 1
            print(f"\nDeleted {removed} duplicate image(s), kept 1 per group.")

    return output_dir

def download_all(project_ids: list[str], **kwargs):

    kwargs.pop("folder_name", None)  # each project gets its own folder name

    results = {}
    total_projects = len(project_ids)

    for i, project_id in enumerate(project_ids, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{total_projects}] Downloading project: {project_id}")
        print(f"{'=' * 60}")

        try:
            if validate:
              input("Begin")
            project = rf.project(project_id)
            output_dir = download_from_project(project, **kwargs)
            results[project_id] = output_dir

        except Exception as e:
            print(f"Failed to download project '{project_id}': {e}")
            results[project_id] = None

    print(f"\n{'=' * 60}")
    print("All downloads finished.")
    for project_id, output_dir in results.items():
        status = output_dir if output_dir else "FAILED"
        print(f"  {project_id}: {status}")

    return results


if __name__ == "__main__":
    # List every project id you want to download here.
    PROJECT_IDS = [
        "pores_datasets-e3gmv-ay73j", # corrupted?
        "pores-a7nzb-ygar9",          # corrupted?
    ]
    download_all(PROJECT_IDS, verbose=False, progress_interval_percentage=10, validate=True)
    