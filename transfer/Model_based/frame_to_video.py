#!/usr/bin/env python3
import glob
import imageio
import shutil
from pathlib import Path
from datetime import datetime

def frames_to_video(
    frames_dir: str = "videos/frames",
    output_dir: str = "transfer/videos",
    output_name: str = None,
    fps: int = 200
):
    """
    Reads all PNG frames in `frames_dir` (relative to CWD),
    stitches them into an MP4 at `fps`,
    writes the video to `output_dir/output_name`,
    and then deletes the source frames.
    """
    # decide on output filename if not given
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"run_video_{timestamp}.mp4"
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_path = output_dir / output_name

    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return

    # ensure the project-relative output folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # gather & sort frames
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"[ERROR] No .png files found in {frames_dir}")
        return

    print(f"[INFO] Writing {len(frames)} frames → {output_path} @ {fps} FPS")
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
    for frame in frames:
        img = imageio.imread(frame)
        writer.append_data(img)
    writer.close()
    print(f"[INFO] Saved video: {output_path}")

    # cleanup frames
    for frame in frames:
        frame.unlink()
    try:
        frames_dir.rmdir()  # remove the now-empty folder
    except OSError:
        pass
    print(f"[INFO] Deleted frames in {frames_dir}")

if __name__ == "__main__":
    frames_to_video()
