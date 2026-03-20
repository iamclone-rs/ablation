import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_config_across import ACROSS_DATASET_SEED, ACROSS_UNSEEN_CLASSES
from src.infer_utils import add_infer_args, run_inference


def main():
    parser = argparse.ArgumentParser(description="Across-dataset inference on QuickDraw from a checkpoint path")
    add_infer_args(parser=parser, dataset_name="quickdraw")
    args = parser.parse_args()

    print(f"Using across-dataset class subset from seed {ACROSS_DATASET_SEED}.")
    run_inference(
        args,
        p_at_k=100,
        allowed_classnames=ACROSS_UNSEEN_CLASSES["quickdraw"],
    )


if __name__ == "__main__":
    main()
