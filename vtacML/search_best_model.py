"""Script to search best model based on config file"""

import logging
import sys
from .pipeline import VTACMLPipe

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)s - %(filename)s:%(lineno)d : "
        "%(message)s",
    )

    vtac_ml = VTACMLPipe(config_file="config/config_seq_0.yaml")

    vtac_ml.train(save_all_model=True, save_path="output/models/seq_0/", verbose=10, n_jobs=5)
