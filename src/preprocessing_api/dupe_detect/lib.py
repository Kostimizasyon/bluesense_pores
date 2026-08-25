import os
import shutil
import imagehash
from PIL import Image
from itertools import combinations
from tqdm import tqdm

# claude oneshot while trying to ask if a spesific function would work, 
# gotta look into it more at some point but it will do for now

# i was just gonna try to go for a cv2 approach just for fun even though it would be quite easier than hashing
# to get a falsepositive dupe due to my spesific approach that i was very much aware of just wanted to do it for fun

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_duplicates(dir_path: str, hash_size: int = 8, threshold: int = 5):
    """Finds near-duplicate images in a directory using perceptual hashing.

    Args:
        dir_path: Directory containing images to scan.
        hash_size: pHash grid size — larger = more sensitive, slower.
        threshold: Max Hamming distance between hashes to count as a
            duplicate. Lower = stricter (fewer false positives, may miss
            some real dupes). 5 is a reasonable starting point for
            hash_size=8 (64-bit hash).

    Returns:
        dict mapping cluster_id -> list of filenames in that cluster.
        Clusters of size 1 are unique images (no duplicates found).
    """
    filenames = sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    )

    hashes = []
    for f in tqdm(filenames, desc="Hashing images"):
        img = Image.open(os.path.join(dir_path, f))
        hashes.append(imagehash.phash(img, hash_size=hash_size))

    n = len(filenames)
    uf = UnionFind(n)

    for i, j in tqdm(list(combinations(range(n), 2)), desc="Comparing pairs"):
        if hashes[i] - hashes[j] <= threshold:
            uf.union(i, j)

    clusters = {}
    for i, f in enumerate(filenames):
        root = uf.find(i)
        clusters.setdefault(root, []).append(f)

    return clusters


def print_report(clusters: dict, verbose : bool = False):
    dupe_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    total_dupes = sum(len(v) - 1 for v in dupe_clusters.values())

    if verbose:
        print("\nDuplicate groups found:")
        for i, (root, files) in enumerate(dupe_clusters.items()):
            print(f"Group {i+1}: {files}")

    print(f"\nTotal images: {sum(len(v) for v in clusters.values())}")
    print(f"Duplicate groups found: {len(dupe_clusters)}")
    print(f"Redundant images (would be removed, keeping 1 per group): {total_dupes}\n")

    return total_dupes

def detect_and_delete_dupes(dir_path: str, threshold: int = 5, validate : bool = False):

    DEDUPED_DIR = os.path.join(dir_path, "deduped")
    os.makedirs(DEDUPED_DIR, exist_ok=True)

    for filename in os.listdir(dir_path):
        source_path = os.path.join(dir_path, filename)
        if (
            os.path.isfile(source_path)
            and os.path.splitext(filename)[1].lower() in VALID_EXTS
        ):
            shutil.copy2(source_path, os.path.join(DEDUPED_DIR, filename))

    clusters = find_duplicates(DEDUPED_DIR, threshold=threshold)
    total_dupes = print_report(clusters)

    if len(clusters) > 0 and total_dupes > 0:

        if validate:
         input(f"\nFound {total_dupes} duplicate images. Press Enter to delete them (keeping 1 per group), or Ctrl+C to cancel...")

        removed = 0
        for files in clusters.values():
            for f in files[1:]:
                os.remove(os.path.join(DEDUPED_DIR, f))
                removed += 1
        print(f"\nDeleted {removed} duplicate image(s), kept 1 per group.")

if __name__ == "__main__":
    path = r"C:/raw_images/"
    clusters = detect_and_delete_dupes(path, threshold=1, validate=True)