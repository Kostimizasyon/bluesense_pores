import os
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


def print_report(clusters: dict):
    dupe_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    total_dupes = sum(len(v) - 1 for v in dupe_clusters.values())

    print(f"\nTotal images: {sum(len(v) for v in clusters.values())}")
    print(f"Duplicate groups found: {len(dupe_clusters)}")
    print(f"Redundant images (would be removed, keeping 1 per group): {total_dupes}\n")

    for i, (root, files) in enumerate(dupe_clusters.items()):
        print(f"Group {i+1}: {files}")

if __name__ == "__main__":
    TEST_DIR = "src/download_dataset/raw_images/pores1.5-2"

    clusters = find_duplicates(TEST_DIR)
    print_report(clusters)