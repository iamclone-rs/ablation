import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_config import UNSEEN_CLASSES
from src.infer_utils import add_infer_args, run_gzs_inference

GZS_TABLE2_CONFIG = {
    "sketchy_2": {"p_at_k": 200, "map_at_k": 200},
    "tuberlin": {"p_at_k": 100, "map_at_k": 0},
}


def main():
    parser = argparse.ArgumentParser(description="GZS-SBIR inference following Table 2 of the SpLIP paper")
    add_infer_args(
        parser=parser,
        dataset_name="tuberlin",
        dataset_choices=sorted(GZS_TABLE2_CONFIG.keys()),
    )
    args = parser.parse_args()

    metric_cfg = GZS_TABLE2_CONFIG[args.dataset]
    print("Running GZS-SBIR inference with unseen sketch queries and seen+unseen photo gallery.")
    run_gzs_inference(
        args=args,
        unseen_classnames=UNSEEN_CLASSES[args.dataset],
        p_at_k=metric_cfg["p_at_k"],
        map_at_k=metric_cfg["map_at_k"],
    )


if __name__ == "__main__":
    main()
