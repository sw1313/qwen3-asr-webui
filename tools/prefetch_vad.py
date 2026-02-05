from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="预下载/预热 Silero VAD（用于离线部署）")
    parser.add_argument(
        "--mode",
        choices=["hub", "local"],
        default="hub",
        help="hub: 使用 torch.hub 自动下载并缓存；local: 使用本地目录加载（用于验证离线包是否可用）",
    )
    parser.add_argument(
        "--local-repo-dir",
        default=os.getenv("VAD_REPO_DIR", ""),
        help="silero-vad 本地目录（mode=local 时必填；也可用环境变量 VAD_REPO_DIR）",
    )
    args = parser.parse_args()

    if args.mode == "local":
        p = Path(args.local_repo_dir).expanduser()
        if not p.exists():
            raise SystemExit(f"本地目录不存在：{p}")
        repo = str(p)
    else:
        repo = "snakers4/silero-vad"

    print(f"[prefetch] loading silero-vad from: {repo}")
    model, utils = torch.hub.load(repo_or_dir=repo, model="silero_vad", force_reload=False, onnx=False)
    _ = model
    _ = utils
    print("[prefetch] ok")


if __name__ == "__main__":
    main()

