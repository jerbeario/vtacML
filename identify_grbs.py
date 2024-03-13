import logging
from pathlib import Path
from pprint import pformat
import time

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits

from vtac.processor import create_valid_sequences

import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format="%(levelname)-8s : %(message)s",
)

# Define directories
# VT_SIM_DIR = Path(os.environ["VT_SIM_DIR"])
# SVOM_DIR = Path(os.environ["SVOM_DIR"])
SVOM_DIR = Path('~/SVOM_pipeline/').expanduser()
VT_SIM_DIR = SVOM_DIR/'VT_simulations'

log = logging.getLogger(__name__)


def get_GRB_ids(qsrclist_vt_fname, GRB_pos=(250, 250), pos_err=(1, 1)):

    ids = {}
    X_GRB, Y_GRB = GRB_pos
    dx, dy = pos_err

    hdul = fits.open(qsrclist_vt_fname)
    for hdu in hdul:
        # Skip Primary HDU
        if "PRIMARY" in hdu.name.upper():
            continue
        seq = hdu.name.split("_")[-1]
        tab = Table.read(hdu)
        cond = (np.abs(tab["X"]-X_GRB) <= dx) & (np.abs(tab["Y"]-Y_GRB) <= dy)

        if len(tab[cond]) == 0:
            log.debug(f"In sequence {seq}: no sources found at GRB position: {GRB_pos}")
        elif len(tab[cond]) == 1:
            ids["OBJID_"+seq] = tab[cond]["OBJID"]
            log.debug(f"In sequence {seq}: one source found at GRB position: {GRB_pos}")
        else:
            log.warning(
                f"In sequence {seq}: more than one sources at GRB position: "
                f"{GRB_pos} +/- {pos_err}:\n{tab[cond]}"
            )
    return ids


if __name__ == '__main__':
    # From Yulei GRB position (X,Y)
    GRB_pos = {
        "bright_case1": [360.50, 291.50],
        "bright_case1a": [346.50, 321.50],
        "bright_case2": [402.50, 219.50],
        "bright_case3": [402.50, 219.50],
        "bright_case4": [377.50, 206.50],
    }

    # Set up
    cases = [
        "bright_case1",
        "bright_case1a",
        "bright_case2",
        "bright_case3",
        "bright_case4",
        # "faint_case1",
        # "faint_case2",
        # "faint_case3",
        # "faint_case4",
    ]
    sequences = ["s1", "s2", "s3", "s4"]
    bands = ["B", "R"]
    sim_ids = [f"{i:07}" for i in np.arange(1, 365)]

    valid_sequences = create_valid_sequences()

    n_sim = n_not_unique = n_no_GRB = n_no_qpo = n_no_sim = 0

    # for case in cases[:1]:
    for case in cases:
        t1 = time.time()
        log.info(f"Starting processing of {case}")

        for sim_id in sim_ids:
            log.info(f"Processing simulation {sim_id}")

            qsrclist_vt_fname = VT_SIM_DIR / f"{case}/fits/qsrclist_vt/{sim_id}_qsrclist_vt.fits"
            qpo_vt_fname = VT_SIM_DIR / f"{case}/fits/qpo_vt/{sim_id}_qpo_vt.fits"

            if qsrclist_vt_fname.exists():
                qsrclist_ids = get_GRB_ids(
                    qsrclist_vt_fname,
                    GRB_pos=GRB_pos[case],
                    pos_err=(1, 1)
                )
                # If no IDs, skip
                if not qsrclist_ids:
                    log.warning(
                        f"No GRB identified for simulation {sim_id} of case {case}. Skipping."
                    )
                    n_no_GRB += 1
                    continue
            else:
                log.warning(f"No QSRCLIST_VT for simulation {sim_id} of {case}. Skipping.")
                n_no_sim += 1
                continue

            # Check file exists
            if qpo_vt_fname.exists():
                _tab = Table.read(qpo_vt_fname, hdu='COMBINED')
            else:
                log.warning(f"QSRCLIST_VT exists but no QPO_VT for simulation {sim_id} of {case}. Skipping.")
                n_no_qpo += 1
                continue

            # Find the unique VT_ID corresponding to the qsrclist_ids
            vt_ids = {}
            for k, v in qsrclist_ids.items():
                vt_id = _tab[_tab[k] == v]["VT_ID"]
                if len(vt_id) > 1:
                    raise ValueError("There should be only one VT_ID per OBJID")
                vt_ids[k] = vt_id[0]

            # Make sure ID is unique
            unique_id = set(vt_ids.values())
            if len(unique_id) > 1:
                log.warning(f"More than one VT_ID found:\n{pformat(vt_ids)} Skipping.")
                n_not_unique += 1
                continue

            # Create column identifying the GRB
            _tab["IS_GRB"] = np.zeros(len(_tab), dtype=int)
            for uid in unique_id:
                _tab["IS_GRB"][_tab["VT_ID"] == uid] = 1

            # Remove unwanted columns
            _tab.remove_columns(
                ["RADEC_OG", "IN_MXT"]
                + ["OBJID_"+seq for seq in valid_sequences]
                + ["RA_"+seq for seq in valid_sequences]
                + ["DEC_"+seq for seq in valid_sequences]
                + ["XFLAG_"+seq for seq in valid_sequences]
                + ["VFLAG_"+seq for seq in create_valid_sequences(seq_num=["1", "2", "3"])]
                # + ["NEW_SRC", "MAG_VAR", "DMAG_CAT"]
            )
            # Add sim_id column
            _tab["CASE"] = np.array([case for i in range(len(_tab))])
            _tab["SIM_ID"] = int(sim_id) * np.ones(len(_tab), dtype=int)

            # Create initial table
            if 'tab' not in locals().keys():
                tab = _tab
            else:
                tab = vstack([tab, _tab], metadata_conflicts='silent')
            n_sim += 1

        t2 = time.time()
        log.info(f"Finished {case} in {t2-t1} seconds")

    log.info(f"Succesfully processed {n_sim} simulations")
    log.warning(
        f"Failed to process {n_not_unique+n_no_GRB+n_no_qpo} simulations:\n"
        f"{n_no_GRB} because no GRB was found at the expected position in QSRCLIST_VT\n"
        f"{n_no_qpo} because QPO_VT did not exist\n"
        f"{n_not_unique} because sequence IDs corresponded to multiple VT_IDs\n"
        f"({n_no_sim} skipped because no simulation)"
    )

    log.info("Final combined table:")
    tab.info()

    # tab.write(VT_SIM_DIR/'combined_qpo_vt.parquet', format='parquet')
    tab.to_pandas().to_parquet(VT_SIM_DIR/'combined_qpo_vt_with_GRB_with_flags.parquet')
