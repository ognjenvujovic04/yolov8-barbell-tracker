"""
Build the video registry datasets/work/videos.csv for the v2 dataset pipeline.

One row per source video in ../data: metadata, content crop (black bars removed),
orientation, intended use and a duplicate group. Videos in one group are exact or
near-duplicate copies (e.g. a clip trimmed from a longer video) and must always land
in the same split. Re-running keeps manual edits of the use/view/split columns.

Run from the project root in WSL:
    python3 build_index.py
    python3 build_index.py --sheets   # also write contact sheets for the clip triage
"""

import argparse
import hashlib
import pickle
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

from dataset_utils import (
    BAR_THRESH, RAW_DIR, SOURCES, VIDEO_EXTS, VIDEOS_CSV, WORK_DIR,
    content_box, dhash, natural_key, orientation, parse_crop, read_videos, split_source,
    video_id, write_videos,
)

SCAN_CACHE = WORK_DIR / "scan_cache.pkl"
TRIAGE_DIR = WORK_DIR / "triage"
SHEET_USES = {"pos", "triage"}

# archive test exercises that use a regular barbell, need a per-clip look, or are excluded
POS_EXERCISES = {
    "bench press", "incline bench press", "decline bench press",
    "squat", "deadlift", "romanian deadlift", "hip thrust",
}
TRIAGE_EXERCISES = {"barbell biceps curl", "shoulder press"}
SKIP_EXERCISES = {"t bar row"}  # landmine: the plate looks identical, but it isn't a regular barbell

HASH_FPS = 1.0  # frames per second hashed for near-duplicate search
MAX_HASHES = 120
MIN_THUMB_STD = 8.0  # skip nearly uniform frames (fades, black frames), they match everything
NEAR_DUP_BITS = 10  # max Hamming distance between two frame hashes (fast candidate search)
NEAR_DUP_MAD = 6.0  # max mean abs difference of 16x16 colour thumbnails (confirms a candidate)
NEAR_DUP_SHARE = 0.5  # share of the shorter video's frames that must match to link two videos
THUMB = 16


def file_md5(path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def scan_video(rel: str) -> dict:
    """Read metadata, content crop, frame hashes and colour thumbnails of one video."""
    path = RAW_DIR / rel
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if not w or not h:
        raise RuntimeError(f"Cannot read video: {path}")

    dur = n / fps if fps else 0.0
    count = int(min(MAX_HASHES, max(5, dur * HASH_FPS)))
    col_bright, row_bright, thumbs = np.zeros(w), np.zeros(h), []
    for f in np.linspace(0, max(n - 1, 0), count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        bright = frame.max(axis=2) > BAR_THRESH
        col_bright += bright.any(axis=0)
        row_bright += bright.any(axis=1)
        thumbs.append(cv2.resize(frame, (320, max(1, round(320 * h / w))), interpolation=cv2.INTER_AREA))
    cap.release()

    crop = content_box(col_bright / len(thumbs), row_bright / len(thumbs)) if thumbs else (0, 0, w, h)
    sx, sy = 320 / w, (thumbs[0].shape[0] / h) if thumbs else 1
    hashes, colors = [], []
    for thumb in thumbs:
        inner = thumb[int(crop[1] * sy):max(int(crop[3] * sy), int(crop[1] * sy) + 1),
                      int(crop[0] * sx):max(int(crop[2] * sx), int(crop[0] * sx) + 1)]
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        if gray.size and gray.std() >= MIN_THUMB_STD:
            hashes.append(dhash(gray))
            colors.append(cv2.resize(inner, (THUMB, THUMB), interpolation=cv2.INTER_AREA))

    return {
        "rel": rel, "w": w, "h": h, "fps": round(fps, 3), "frames": n, "dur": round(dur, 2),
        "crop": list(crop), "md5": file_md5(path), "hashes": hashes,
        "colors": np.array(colors, dtype=np.uint8).reshape(-1, THUMB * THUMB * 3),
    }


def find_videos() -> list[str]:
    rels = []
    for folder in SOURCES:
        base = RAW_DIR / folder
        for p in base.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                rels.append(p.relative_to(RAW_DIR).as_posix())
    return sorted(set(rels), key=natural_key)


def scan_all(rels: list[str], workers: int) -> dict[str, dict]:
    """Scan videos, reusing cached results for files whose size and mtime are unchanged."""
    cache = pickle.loads(SCAN_CACHE.read_bytes()) if SCAN_CACHE.exists() else {}
    stamp = {rel: f"{(RAW_DIR / rel).stat().st_size}:{int((RAW_DIR / rel).stat().st_mtime)}" for rel in rels}
    todo = [rel for rel in rels if cache.get(rel, {}).get("stamp") != stamp[rel]]
    print(f"Scanning {len(todo)} of {len(rels)} videos ({len(rels) - len(todo)} cached)...")
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(scan_video, todo), 1):
            result["stamp"] = stamp[result["rel"]]
            cache[result["rel"]] = result
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
    cache = {rel: cache[rel] for rel in rels}
    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE.write_bytes(pickle.dumps(cache))
    return cache


def near_duplicate_pairs(ids: list[str], scans: dict[str, dict]) -> set[tuple[str, str]]:
    """
    Pairs of videos where at least NEAR_DUP_SHARE of one video's frames have a near-identical
    frame in the other, e.g. a clip trimmed from a longer video or a re-encoded copy.

    Candidates come from the 64-bit frame hashes; each is confirmed on colour thumbnails,
    because the gradient hash alone can't tell apart different people on a plain background.
    """
    owner = np.array([i for i, vid in enumerate(ids) for _ in scans[vid]["hashes"]], dtype=np.int32)
    values = np.array([h for vid in ids for h in scans[vid]["hashes"]], dtype=np.uint64)
    colors = np.concatenate([scans[vid]["colors"] for vid in ids]).astype(np.int16)
    if not len(values):
        return set()
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    matched = defaultdict(set)  # (video a, video b) -> frames of a that match b
    for start in range(0, len(values), 256):
        block = values[start:start + 256]
        xor = np.bitwise_xor(block[:, None], values[None, :])
        dist = popcount[xor.view(np.uint8)].reshape(len(block), len(values), 8).sum(axis=2)
        rows, cols = np.nonzero(dist <= NEAR_DUP_BITS)
        rows = rows + start
        keep = owner[rows] != owner[cols]
        rows, cols = rows[keep], cols[keep]
        mad = np.abs(colors[rows] - colors[cols]).mean(axis=1)
        for r, c in zip(rows[mad <= NEAR_DUP_MAD], cols[mad <= NEAR_DUP_MAD]):
            matched[(owner[r], owner[c])].add(r)
    pairs = set()
    for (a, b), frames in matched.items():
        n = len(scans[ids[a]]["hashes"])
        if len(frames) >= max(2, NEAR_DUP_SHARE * n) or (n == 1 and frames):
            pairs.add(tuple(sorted((ids[a], ids[b]))))
    return pairs


def default_use(source: str, exercise: str) -> str:
    if source == "nr":
        return "hard"
    if source in ("atr", "att"):
        if exercise in POS_EXERCISES:
            return "pos"
        if exercise in TRIAGE_EXERCISES:
            return "triage"
        if exercise in SKIP_EXERCISES:
            return "skip"
        return "neg"
    return "pos"


def build_rows(scans: dict[str, dict]) -> list[dict]:
    rows = {}
    for rel, s in scans.items():
        vid = video_id(rel)
        assert vid not in rows, f"Duplicate video id {vid}: {rel} / {rows[vid]['path']}"
        source, inner = split_source(rel)
        parts = inner.split("/")
        exercise = parts[-2].lower().strip() if source in ("ar", "atr", "att") and len(parts) > 1 else ""
        x0, y0, x1, y1 = s["crop"]
        rows[vid] = {
            "video_id": vid, "path": rel, "source": source, "exercise": exercise,
            "w": s["w"], "h": s["h"], "fps": s["fps"], "frames": s["frames"], "dur": s["dur"],
            "crop": f"{x0} {y0} {x1} {y1}",
            "orientation": orientation(x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else "",
            "md5": s["md5"], "use": default_use(source, exercise), "view": "", "split": "",
        }

    # Union-find over exact copies (same md5) and near-duplicates (matching frame hashes)
    parent = {vid: vid for vid in rows}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=natural_key)] = min(ra, rb, key=natural_key)

    by_md5 = defaultdict(list)
    for vid, r in rows.items():
        by_md5[r["md5"]].append(vid)
    for vids in by_md5.values():
        for other in vids[1:]:
            union(vids[0], other)

    ids = sorted(rows, key=natural_key)
    for a, b in near_duplicate_pairs(ids, {vid: scans[rows[vid]["path"]] for vid in ids}):
        union(a, b)

    members = defaultdict(list)
    for vid in ids:
        members[find(vid)].append(vid)
    source_rank = {prefix: i for i, prefix in enumerate(["d1", "tt", "atr", "att", "ar", "nr"])}
    for group, vids in members.items():
        for vid in vids:
            rows[vid]["group"] = group
        # Exact copies and copies from another source (archive clips that dataset1 was made from,
        # TikTok originals of dataset1 clips, archive test/test copies of raw_data) add no new
        # footage: keep one primary video. Near-duplicates within a source are separate takes.
        primary = min(vids, key=lambda v: (source_rank[rows[v]["source"]], natural_key(v)))
        for vid in vids:
            if vid == primary:
                continue
            exact = rows[vid]["md5"] == rows[primary]["md5"]
            if exact or rows[vid]["source"] != rows[primary]["source"]:
                rows[vid]["use"] = "dup"
    return list(rows.values())


def merge_manual_edits(rows: list[dict]) -> list[dict]:
    """
    Keep manual edits from an existing registry: view, split and use (e.g. triage -> pos).
    'dup' is never kept from before, it is always recomputed from the duplicate groups.
    """
    if not VIDEOS_CSV.exists():
        return rows
    old = {r["video_id"]: r for r in read_videos()}
    for r in rows:
        prev = old.get(r["video_id"])
        if not prev:
            continue
        for key in ("view", "split"):
            if prev.get(key):
                r[key] = prev[key]
        if prev.get("use") not in ("", "dup") and r["use"] != "dup":
            r["use"] = prev["use"]
    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n{len(rows)} videos, {len({r['group'] for r in rows})} groups -> {VIDEOS_CSV}")
    table = Counter((r["source"], r["orientation"], r["use"]) for r in rows)
    minutes = defaultdict(float)
    for r in rows:
        minutes[(r["source"], r["orientation"], r["use"])] += float(r["dur"]) / 60
    print(f"\n{'source':6} {'orient':6} {'use':7} {'videos':>6} {'minutes':>8}")
    for key in sorted(table):
        print(f"{key[0]:6} {key[1]:6} {key[2]:7} {table[key]:6d} {minutes[key]:8.1f}")
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r["video_id"])
    multi = {g: v for g, v in groups.items() if len(v) > 1}
    print(f"\n{len(multi)} groups with more than one video:")
    for g in sorted(multi, key=natural_key):
        print(f"  {g}: {', '.join(sorted(multi[g], key=natural_key))}")


def contact_sheet(row: dict) -> None:
    """Save a 3x2 grid of frames (cropped to the content) for deciding a clip's view and use."""
    x0, y0, x1, y1 = parse_crop(row["crop"])
    cap = cv2.VideoCapture(str(RAW_DIR / row["path"]))
    n = int(row["frames"])
    tiles = []
    for f in np.linspace(0.05 * n, 0.95 * max(n - 1, 0), 6).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        tile = frame[y0:y1, x0:x1] if ok else np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        tiles.append(cv2.resize(tile, (round(240 * tile.shape[1] / tile.shape[0]), 240)))
    cap.release()
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    header = np.full((32, grid.shape[1], 3), 255, np.uint8)
    title = f"{row['video_id']}  use={row['use']}  {row['exercise'] or row['source']}  {row['orientation']}  {row['dur']}s"
    cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(TRIAGE_DIR / f"{row['video_id']}.jpg"), np.vstack([header, grid]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_sheets(rows: list[dict], workers: int) -> None:
    todo = [r for r in rows if r["use"] in SHEET_USES and not (TRIAGE_DIR / f"{r['video_id']}.jpg").exists()]
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        pool.map(contact_sheet, todo)
    total = sum(1 for r in rows if r["use"] in SHEET_USES)
    print(f"\nContact sheets: {len(todo)} new, {total} total -> {TRIAGE_DIR}")
    print("Triage in videos.csv: view=front for front/back views; triage -> pos (straight loaded barbell) or skip.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasets/work/videos.csv from the videos in ../data.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel video readers (default: 6)")
    parser.add_argument("--sheets", action="store_true",
                        help="Also write contact sheets to datasets/work/triage/ for pos and triage videos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rels = find_videos()
    scans = scan_all(rels, args.workers)
    rows = merge_manual_edits(build_rows(scans))
    write_videos(rows)
    print_summary(rows)
    if args.sheets:
        write_sheets(rows, args.workers)


if __name__ == "__main__":
    main()
