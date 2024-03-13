import time
import logging
import sys
from datetime import timedelta
from pathlib import Path
from logging.config import dictConfig
import os
import argparse

import numpy as np
import pandas as pd
from astropy.table import Table, vstack
# from multiprocessing import Pool
# import multiprocessing

# from svom.utils import FitsProductFromModel
from vtac.processor import create_valid_sequences

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(funcName)s - %(filename)s:%(lineno)d : %(message)s",
)

# Define directories
VT_SIM_DIR = Path(os.environ["VT_SIM_DIR"])
SVOM_DIR = Path(os.environ["SVOM_DIR"])

log = logging.getLogger(__name__)


if __name__ == '__main__':
    # Set up
    cases = [
        "bright_case1",
        "bright_case1a",
        "bright_case2",
        "bright_case3",
        "bright_case4",
        "faint_case1",
        "faint_case2",
        "faint_case3",
        "faint_case4",
    ]
    sequences = ["s1", "s2", "s3", "s4"]
    bands = ["B", "R"]
    sim_ids = [f"{i:07}" for i in np.arange(1, 365)]

    # case = 'bright_case1'
    valid_sequences = create_valid_sequences()
    for case in cases:
        for sim_id in sim_ids:
            qpo_vt_fname = VT_SIM_DIR / f"{case}/fits/qpo_vt/{sim_id}_qpo_vt.fits"

            # Check file exists
            if qpo_vt_fname.exists():
                _tab = Table.read(qpo_vt_fname, hdu='COMBINED')
            else:
                log.warning(f"No QPO_VT for simulation {sim_id} of {case}. Skipping.")
                continue

            # Remove unwanted columns
            _tab.remove_columns(
                ["RADEC_OG", "IN_MXT"]
                + ["OBJID_"+seq for seq in valid_sequences]
                + ["RA_"+seq for seq in valid_sequences]
                + ["DEC_"+seq for seq in valid_sequences]
                + ["XFLAG_"+seq for seq in valid_sequences]
                + ["VFLAG_"+seq for seq in create_valid_sequences(seq_num=["1", "2", "3"])]
                + ["NEW_SRC", "MAG_VAR", "DMAG_CAT"]
            )
            # Add sim_id column
            _tab["CASE"] = np.array([case for i in range(len(_tab))])
            _tab["SIM_ID"] = int(sim_id) * np.ones(len(_tab), dtype=int)
            _tab["IS_GRB"] = np.zeros(len(_tab), dtype=bool)

            # Create initial table
            if 'tab' not in locals().keys():
                tab = _tab
            else:
                tab = vstack([tab, _tab], metadata_conflicts='silent')

    log.info(f"Final combined table: {tab.info()}")

    # tab.write(VT_SIM_DIR/'combined_qpo_vt.parquet', format='parquet')
    tab.to_pandas().to_parquet(VT_SIM_DIR/'combined_qpo_vt.parquet')
