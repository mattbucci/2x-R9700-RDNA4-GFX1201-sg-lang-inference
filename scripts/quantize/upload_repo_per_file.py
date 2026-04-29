#!/usr/bin/env python3
"""Upload an HF repo one file per commit, skipping files already present
server-side. Works around hf-cli atomic commit stalls on multi-shard repos."""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. mattbucci/Qwen3-Coder-Next-REAM-AWQ")
    p.add_argument("--src", required=True, help="local dir to upload")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--prefix", default="", help="optional path prefix in repo")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1

    src = Path(os.path.expanduser(args.src)).resolve()
    if not src.is_dir():
        print(f"src dir not found: {src}", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)

    try:
        existing = set(api.list_repo_files(args.repo, token=args.token))
    except HfHubHTTPError as e:
        print(f"list_repo_files failed: {e}", file=sys.stderr)
        existing = set()
    print(f"existing on server: {len(existing)} files")

    files = sorted(p for p in src.rglob("*") if p.is_file())
    skip_dirs = {".cache", "__pycache__", ".git"}
    files = [p for p in files if not any(part in skip_dirs for part in p.parts)]
    print(f"local files: {len(files)}")

    n_skipped = n_uploaded = n_failed = 0
    for fp in files:
        rel = fp.relative_to(src).as_posix()
        path_in_repo = (args.prefix.rstrip("/") + "/" + rel).lstrip("/") if args.prefix else rel
        if path_in_repo in existing:
            n_skipped += 1
            continue

        size_mb = fp.stat().st_size / 1024 / 1024
        print(f"[upload] {path_in_repo} ({size_mb:.1f} MB)", flush=True)
        if args.dry_run:
            n_uploaded += 1
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=path_in_repo,
                repo_id=args.repo,
                repo_type="model",
                commit_message=f"add {path_in_repo}",
            )
            n_uploaded += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            n_failed += 1

    print(f"\nresult: skipped={n_skipped} uploaded={n_uploaded} failed={n_failed}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
