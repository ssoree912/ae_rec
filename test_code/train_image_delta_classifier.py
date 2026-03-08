import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "training_code" / "train_image_delta_classifier.py"
    print(f"[Redirect] Use training entrypoint: {target}")
    runpy.run_path(str(target), run_name="__main__")
