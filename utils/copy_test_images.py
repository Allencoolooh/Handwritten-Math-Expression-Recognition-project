# tools/copy_test_images.py
from pathlib import Path
import shutil
import os


def main():
    # === 1) 自动定位项目根目录：以脚本所在 tools/ 的上一级为根 ===
    project_root = Path(__file__).resolve().parents[1]
    print("[ProjectRoot]", project_root)
    print("[CWD]", Path.cwd())

    # === 2) 文件路径 ===
    text_path = project_root / "data/test.txt"
    assert text_path.is_file(), f"test.txt not found: {text_path}"

    # 源/目标目录（都以 project_root 为基准）
    src_dir = project_root / "data" / "icdar_raw" / "train_images"
    dst_dir = project_root / "data" / "icdar_raw" / "test_images"
    assert src_dir.is_dir(), f"Source dir not found: {src_dir}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    missing = 0

    with text_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                img_path_str, _ = line.split("\t", 1)
            except ValueError:
                print(f"[Line {line_id}] Invalid format (no TAB), skipped")
                continue

            img_path_str = img_path_str.strip().strip('"').strip("'")

            # === 3) 只取文件名（最稳），避免相对路径根错 ===
            # 兼容 \ 或 /：
            img_name = Path(img_path_str.replace("\\", "/")).name

            src_img = src_dir / img_name
            dst_img = dst_dir / img_name

            if not src_img.is_file():
                print(f"[Missing] {src_img}")
                missing += 1
                continue

            if dst_img.exists():
                skipped += 1
                continue

            shutil.copy2(src_img, dst_img)
            copied += 1

    print("\n===== Copy Finished =====")
    print(f"Copied images : {copied}")
    print(f"Skipped exist : {skipped}")
    print(f"Missing files : {missing}")
    print(f"Source folder : {src_dir}")
    print(f"Target folder : {dst_dir}")


if __name__ == "__main__":
    main()
