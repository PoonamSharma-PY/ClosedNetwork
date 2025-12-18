#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform trivalency closed networks to weighted probabilities and ensure seed_sets.json
is present in every (B,l) folder under the output root.

Features:
- Mirrors the directory structure from --in-root to --out-root.
- For every ClosedNetwork_*.txt, replaces probability with that from --weighted (u v p).
- Handles missing edges via --policy {keep,drop,zero,raise}.
- Copies seed_sets.json from the input folder to the corresponding output folder.
- If seed_sets.json is missing in input, can generate one for ALL (B,l) folders from a
  master seed listing (your tab-separated file with "ClosedNetwork_i" sections).
- Writes a summary CSV if --log is provided.

Usage:
  python transform_and_seedcopy.py \
    --weighted ./euemail_weighted.txt \
    --in-root ./Budget1000 \
    --out-root ./Budget1000Weighted \
    --policy keep \
    --zero-value 0.0 \
    --workers 24 \
    --log ./remap_summary.csv \
    --seed-source ./seeds_master.txt   # (optional fallback; see format below)

Seed source (fallback) format example:
ClosedNetwork_1
0\t184
1\t276
...
ClosedNetwork_2
0\t73
1\t472
...

Blank lines allowed. "ClosedNetwork_i" lines may end with a colon or tabs/spaces.
"""

import os
import sys
import gzip
import json
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# ---------- I/O helpers ----------

def open_maybe_gzip(path, mode="rt"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode=mode, encoding=None if "b" in mode else "utf-8")
    return open(p, mode=mode, encoding=None if "b" in mode else "utf-8")

def detect_delimiter(sample_line: str):
    if "," in sample_line:
        return ","
    if "\t" in sample_line:
        return "\t"
    return None  # whitespace split

def parse_weighted_line(line: str, delim=None):
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split(delim) if delim else s.split()
    if len(parts) < 3:
        return None
    u, v, p = parts[0], parts[1], parts[2]
    try:
        pf = float(p)
    except:
        return None
    return u, v, pf

def parse_edge_line(line: str, delim=None):
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split(delim) if delim else s.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]

# ---------- Weighted master ----------

def load_weighted_map(weighted_path: str) -> Dict[Tuple[str, str], float]:
    edge2prob: Dict[Tuple[str, str], float] = {}
    with open_maybe_gzip(weighted_path, "rt") as f:
        first_data = None
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                first_data = s
                break
        if first_data is None:
            raise ValueError("Weighted file is empty or only comments.")
        delim = detect_delimiter(first_data)
        rec = parse_weighted_line(first_data, delim)
        if rec:
            edge2prob[(rec[0], rec[1])] = rec[2]
        for line in f:
            rec = parse_weighted_line(line, delim)
            if rec:
                edge2prob[(rec[0], rec[1])] = rec[2]
    return edge2prob

# ---------- Seeds parsing (fallback) ----------

def parse_seed_source(seed_source_path: Optional[str]) -> Optional[Dict[str, List[int]]]:
    """
    Parse a master seed text file that looks like:

    ClosedNetwork_1
    0\t184
    1\t276
    ...
    ClosedNetwork_2
    0\t73
    ...

    Returns: {"ClosedNetwork_1": [184, 276, ...], "ClosedNetwork_2": [...], ...}
    """
    if not seed_source_path:
        return None
    p = Path(seed_source_path)
    if not p.exists():
        print(f"[WARN] seed-source not found: {seed_source_path}", file=sys.stderr)
        return None

    result: Dict[str, List[Tuple[int,int]]] = {}
    current_key: Optional[str] = None

    with open(seed_source_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # network header?
            if line.lower().startswith("closednetwork_"):
                # normalize e.g. "ClosedNetwork_1:" -> "ClosedNetwork_1"
                name = line.split()[0].rstrip(":")
                current_key = name
                if current_key not in result:
                    result[current_key] = []
                continue
            # inside a network: expect "idx<TAB/space>node"
            if current_key:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        node = int(parts[1])
                        result[current_key].append((idx, node))
                    except:
                        # ignore malformed lines
                        pass

    # sort by idx and strip idx
    final: Dict[str, List[int]] = {}
    for k, lst in result.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        final[k] = [node for _, node in lst_sorted]
    if not final:
        print(f"[WARN] seed-source parsed empty: {seed_source_path}", file=sys.stderr)
        return None
    return final

# ---------- Core remap ----------

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def remap_one_file(in_file: Path,
                   out_file: Path,
                   edge2prob: Dict[Tuple[str,str], float],
                   policy: str = "keep",
                   zero_value: float = 0.0,
                   output_delim: str = " ") -> dict:
    missing_edges = 0
    kept_edges = 0
    dropped_edges = 0
    total_in = 0

    # detect delim
    with open_maybe_gzip(in_file, "rt") as fin:
        first_data = None
        for line in fin:
            s = line.strip()
            if s and not s.startswith("#"):
                first_data = s
                break
    if first_data is None:
        ensure_parent(out_file)
        out_file.write_text("", encoding="utf-8")
        return {"file": str(in_file), "out_file": str(out_file),
                "total_in": 0, "kept_edges": 0, "dropped_edges": 0,
                "missing_edges": 0, "status": "empty"}
    delim = detect_delimiter(first_data)

    ensure_parent(out_file)
    with open_maybe_gzip(in_file, "rt") as fin, open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = parse_edge_line(line, delim)
            if rec is None:
                continue
            total_in += 1
            u, v = rec
            key = (u, v)
            if key in edge2prob:
                p = edge2prob[key]
                fout.write(f"{u}{output_delim}{v}{output_delim}{p:.12g}\n")
                kept_edges += 1
            else:
                missing_edges += 1
                if policy == "drop":
                    dropped_edges += 1
                    continue
                elif policy in ("zero", "keep"):
                    fout.write(f"{u}{output_delim}{v}{output_delim}{zero_value:.12g}\n")
                    kept_edges += 1
                elif policy == "raise":
                    raise KeyError(f"Missing weighted prob for edge {u} {v} (file: {in_file})")
                else:
                    raise ValueError(f"Unknown policy: {policy}")

    status = "ok"
    if missing_edges > 0:
        status = "ok_dropped_missing" if policy == "drop" else "ok_with_missing"
    return {"file": str(in_file), "out_file": str(out_file),
            "total_in": total_in, "kept_edges": kept_edges,
            "dropped_edges": dropped_edges, "missing_edges": missing_edges,
            "status": status}

def list_closednet_files(in_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Returns:
      - files: list of ClosedNetwork_*.txt paths
      - bl_folders: unique list of (B,l) folders containing those files
    """
    files: List[Path] = []
    bl_folders_set = set()
    for p in in_root.rglob("*"):
        if p.is_file() and p.name.startswith("ClosedNetwork_") and p.suffix.lower() in (".txt", ".csv", ".tsv", ".edges", ".edgelist", ""):
            files.append(p)
            bl_folders_set.add(str(p.parent))
    files.sort()
    bl_folders = [Path(s) for s in sorted(bl_folders_set)]
    return files, bl_folders

def copy_or_write_seeds(in_folder: Path, out_folder: Path, seed_master: Optional[Dict[str, List[int]]]):
    """
    Ensure out_folder/seed_sets.json exists:
      - If in_folder/seed_sets.json exists -> copy its contents.
      - Else if seed_master is provided -> write it as JSON.
      - Else -> warn once.
    """
    src = in_folder / "seed_sets.json"
    dst = out_folder / "seed_sets.json"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return "copied"
        except Exception as e:
            print(f"[WARN] Failed to read {src}: {e}. Falling back to seed_master.", file=sys.stderr)

    if seed_master is not None:
        try:
            dst.write_text(json.dumps(seed_master, indent=2), encoding="utf-8")
            return "written_from_master"
        except Exception as e:
            print(f"[WARN] Failed to write {dst}: {e}", file=sys.stderr)
            return "failed"
    else:
        print(f"[WARN] seed_sets.json missing in {in_folder} and no --seed-source provided.", file=sys.stderr)
        return "missing"

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Remap closed networks to weighted probabilities and ensure seed_sets.json in every (B,l) folder.")
    ap.add_argument("--weighted", required=True, help="Path to master weighted cascade file (u v p) [.gz ok].")
    ap.add_argument("--in-root", required=True, help="Root containing Budget*/Closed_Network_Outputs_l*_B*_x*/ClosedNetwork_*.txt")
    ap.add_argument("--out-root", required=True, help="Output root; structure mirrors in-root.")
    ap.add_argument("--policy", default="keep", choices=["keep","drop","zero","raise"], help="Missing-edge policy. Default: keep (prob=zero-value).")
    ap.add_argument("--zero-value", type=float, default=0.0, help="Probability to write when missing and policy is keep/zero.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers. Default: all CPUs.")
    ap.add_argument("--output-delim", default=" ", help="Delimiter for output files. Default: space.")
    ap.add_argument("--log", default=None, help="Optional CSV summary path.")
    ap.add_argument("--seed-source", default=None, help="Optional master seed file (tab-separated with 'ClosedNetwork_i' sections). Used if a folder lacks seed_sets.json.")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading weighted map from: {args.weighted}")
    edge2prob = load_weighted_map(args.weighted)
    print(f"[INFO] Weighted edges loaded: {len(edge2prob):,}")

    print(f"[INFO] Scanning input tree: {in_root}")
    files, bl_folders = list_closednet_files(in_root)
    if not files:
        print(f"[ERROR] No ClosedNetwork_*.txt files found under: {in_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Closed networks to process: {len(files):,}")
    print(f"[INFO] (B,l) folders found: {len(bl_folders):,}")

    # Parse seed master (fallback) if provided
    seed_master = parse_seed_source(args.seed_source) if args.seed_source else None
    if seed_master is not None:
        # quick sanity: ensure at least ClosedNetwork_1 exists in mapping
        any_key = next(iter(seed_master), None)
        print(f"[INFO] seed-source parsed: {len(seed_master)} entries (e.g., {any_key})")

    # Ensure seeds exist per (B,l) folder in OUTPUT tree
    seed_stats = {"copied":0, "written_from_master":0, "missing":0, "failed":0}
    for in_folder in bl_folders:
        # Map input folder to output folder
        rel = in_folder.relative_to(in_root)
        out_folder = out_root / rel
        status = copy_or_write_seeds(in_folder, out_folder, seed_master)
        seed_stats[status] = seed_stats.get(status, 0) + 1
    print(f"[INFO] seed_sets.json status → copied: {seed_stats.get('copied',0)}, "
          f"written_from_master: {seed_stats.get('written_from_master',0)}, "
          f"missing: {seed_stats.get('missing',0)}, failed: {seed_stats.get('failed',0)}")

    # Prepare worker
    def make_out_path(in_path: Path) -> Path:
        rel = in_path.relative_to(in_root)
        return out_root / rel

    worker = partial(
        remap_one_file,
        edge2prob=edge2prob,
        policy=args.policy,
        zero_value=args.zero_value,
        output_delim=args.output_delim
    )

    results = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut2src = {}
            for f in files:
                of = make_out_path(f)
                fut = ex.submit(worker, f, of)
                fut2src[fut] = f
            for fut in as_completed(fut2src):
                try:
                    res = fut.result()
                    results.append(res)
                    print(f"[OK] {res['file']} → {res['out_file']} | kept={res['kept_edges']} missing={res['missing_edges']} dropped={res['dropped_edges']}")
                except Exception as e:
                    src = fut2src[fut]
                    print(f"[ERROR] {src}: {e}", file=sys.stderr)
    else:
        for f in files:
            try:
                of = make_out_path(f)
                res = worker(f, of)
                results.append(res)
                print(f"[OK] {res['file']} → {res['out_file']} | kept={res['kept_edges']} missing={res['missing_edges']} dropped={res['dropped_edges']}")
            except Exception as e:
                print(f"[ERROR] {f}: {e}", file=sys.stderr)

    # Optional CSV log
    if args.log:
        try:
            import csv
            with open(args.log, "w", newline="", encoding="utf-8") as fo:
                w = csv.DictWriter(fo, fieldnames=[
                    "file","out_file","total_in","kept_edges","dropped_edges","missing_edges","status"
                ])
                w.writeheader()
                for r in results:
                    w.writerow(r)
            print(f"[INFO] Log written: {args.log}")
        except Exception as e:
            print(f"[WARN] Failed to write log: {e}", file=sys.stderr)

    # Summary
    tot_in = sum(r["total_in"] for r in results)
    tot_kept = sum(r["kept_edges"] for r in results)
    tot_drop = sum(r["dropped_edges"] for r in results)
    tot_miss = sum(r["missing_edges"] for r in results)
    print("\n=== SUMMARY ===")
    print(f"Files processed : {len(results):,}")
    print(f"Edges (input)   : {tot_in:,}")
    print(f"Edges kept      : {tot_kept:,}")
    print(f"Edges dropped   : {tot_drop:,}")
    print(f"Edges missing   : {tot_miss:,} (policy={args.policy})")

if __name__ == "__main__":
    main()
