from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
TRAINING_CODE_DIR = THIS_DIR.parent
if str(TRAINING_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_DIR))


def train_rectifier_main():
    """
    D3-style entry helper.
    Delegates to this repo's rectifier trainer.
    """
    from train_rectifier import main  # noqa: WPS433

    main()

