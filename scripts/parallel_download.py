"""
Multi-connection range-request downloader. Built because single-stream
downloads on this network throttle hard after ~30-40MB (observed: EDM
checkpoints and the first ~30MB of a HF parquet file transferred at
0.5-0.85 MB/s, then decayed to ~0.1 MB/s and stalled -- looks like a
per-connection/per-flow shaping policy). Splitting into N concurrent
Range-request chunks works around per-flow throttling by using N flows.

Usage: python scripts/parallel_download.py <url> <dest_path> [--connections N]
"""
from __future__ import annotations
import argparse
import concurrent.futures
import os
import sys
import time
import urllib.request


def get_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download_chunk(url: str, dest_path: str, start: int, end: int, chunk_id: int, retries: int = 5):
    """Fetch bytes [start, end] (inclusive) and write them at the right
    offset in dest_path. Retries with a fresh connection on stall/error --
    single flows here degrade over time, so a chunk that stalls is worth
    restarting rather than waiting out."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read(end - start + 1)
            with open(dest_path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return chunk_id, len(data)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def parallel_download(url: str, dest_path: str, connections: int = 8):
    total = get_size(url)
    print(f"Total size: {total/1e6:.1f} MB, {connections} connections")
    # Pre-allocate the destination file.
    with open(dest_path, "wb") as f:
        f.truncate(total)

    chunk_size = (total + connections - 1) // connections
    ranges = []
    for i in range(connections):
        start = i * chunk_size
        end = min(start + chunk_size - 1, total - 1)
        if start > end:
            continue
        ranges.append((start, end, i))

    t0 = time.time()
    done_bytes = [0] * len(ranges)
    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as ex:
        futures = {
            ex.submit(download_chunk, url, dest_path, start, end, cid): cid
            for start, end, cid in ranges
        }
        for fut in concurrent.futures.as_completed(futures):
            cid, n = fut.result()
            done_bytes[cid] = n
            got = sum(done_bytes)
            elapsed = time.time() - t0
            print(f"chunk {cid} done ({n/1e6:.1f} MB). total {got/1e6:.1f}/{total/1e6:.1f} MB, "
                  f"{got/1e6/max(elapsed,0.01):.2f} MB/s avg", flush=True)

    actual = os.path.getsize(dest_path)
    print(f"Done: {dest_path} ({actual/1e6:.1f} MB) in {time.time()-t0:.1f}s")
    if actual != total:
        print(f"WARNING: size mismatch, expected {total}, got {actual}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("dest")
    p.add_argument("--connections", type=int, default=8)
    args = p.parse_args()
    parallel_download(args.url, args.dest, args.connections)
