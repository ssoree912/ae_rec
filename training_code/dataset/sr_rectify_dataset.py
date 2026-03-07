from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
TRAINING_CODE_DIR = THIS_DIR.parent
if str(TRAINING_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_DIR))

from rectified.datasets import RealSRPairDataset  # noqa: E402


class SRRectifyDataset(RealSRPairDataset):
    """
    Compatibility wrapper with D3-like file name.
    Internally reuses RealSRPairDataset from this repo.
    """

    pass

