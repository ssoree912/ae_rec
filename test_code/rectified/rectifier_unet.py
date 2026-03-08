from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SRC = THIS_DIR.parent.parent / "training_code" / "rectified" / "rectifier_unet.py"

if not SRC.is_file():
    raise FileNotFoundError(f"Cannot find training rectifier module: {SRC}")

spec = spec_from_file_location("training_rectifier_unet_impl", str(SRC))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from: {SRC}")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)

RectifierUNet = mod.RectifierUNet
load_rectifier = mod.load_rectifier
