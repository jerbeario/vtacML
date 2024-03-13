import time
import logging
from datetime import timedelta
from pathlib import Path
from logging.config import dictConfig
import os
import argparse

import numpy as np
from astropy.table import Table
# from multiprocessing import Pool
# import multiprocessing

# from svom.utils import FitsProductFromModel

from vtac.handler import get_product_template, QuickPositionHandler

# Define directories
VT_SIM_DIR = Path(os.environ["VT_SIM_DIR"])
SVOM_DIR = Path(os.environ["SVOM_DIR"])

log = logging.getLogger(__name__)


def get_cli_args():
    """
    Retrieve command line options.

    Returns
    -------
    parser.parse_args: argparse.Namespace
        Namespace containing each argument with its associated value.
    """

    parser = argparse.ArgumentParser(add_help=True, description="Processor for Yulei's simulations")
    parser.add_argument(
        "case",
        help="Name of the case, e.g. 'bright_case1'",
    )

    return parser.parse_args()


def config_root_logger():

    dictConfig(
        {
            "version": 1,
            # Don't disable existing loggers, otherwise you lose all logs
            # of 3rd party libraries (including the ones from vtac...)
            "disable_existing_loggers": 0,
            "formatters": {
                "root_formatter": {
                    "format": "[%(asctime)s UTC] - %(levelname)s - [%(name)s] - %(filename)s:%(lineno)d - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "root_formatter",
                },
                # If you want to log to a file
                # "log_file": {
                #     "class": "logging.FileHandler",
                #     "level": "DEBUG",
                #     "filename": log_file,
                #     "formatter": "root_formatter",
                #     "mode": "w",
                # },
            },
            "loggers": {
                # Root logger
                "": {
                    "handlers": [
                        # "log_file",
                        "console",
                    ],
                    "level": "DEBUG",
                    "propagate": True,
                },
            },
        }
    )


def add_logging_handler(case, sim_id, level):
    """
    Add a log handler to separate file for current case/sim_id
    """

    # Add logging file
    log_file = VT_SIM_DIR / f"{case}/logs/{level}/{sim_id}.log"
    log_file.parent.mkdir(exist_ok=True)
    log_handler = logging.FileHandler(log_file)#, mode='a')

    # Set logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % level)
    log_handler.setLevel(numeric_level)

    # Set format
    formatter = logging.Formatter(
        "[%(asctime)s UTC] "
        "- %(levelname)s "
        "- [%(name)s] "
        "- %(filename)s:%(lineno)d "
        "- %(message)s"
    )
    log_handler.setFormatter(formatter)

    logging.getLogger().addHandler(log_handler)

    return log_handler


def remove_logging_handler(log_handler):
    # Remove log handler from root logger
    logging.getLogger().removeHandler(log_handler)

    # Close the log handler so that the lock on log file can be released
    log_handler.close()


def read_header(fname):
    hdr = {}
    with open(fname, "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                _split_line = line.strip().split()
                if len(_split_line) == 2:
                    key = _split_line[0]
                    val = _split_line[1]
                    hdr[key.strip("#").upper()] = val
    return hdr


def fill_headers(fits_template, keywords_dict):
    """
    Fill the headers of the various fits extensions with the
    keywords used during processing.

    Parameters
    ----------
    fits_template : FitsProductFromModel
        FitsProductFromModel whose keywords need to be updated
    keywords_dict : dict
        Dictionary whose keys are the name of the extension
        and value is a dictionary of keywords. Example:
        keywords_dict={'PrimaryHDU':{'OBS_ID':'sb12345678'},'R1':{'BAND':'VT_R'}}
    """

    # Iterate over the various fits extensions
    for ext, keywords in keywords_dict.items():
        log.debug("Processing keywords for extension '%s'" % ext)


def reformat_qsrclist(qsrclist_vt, sim_id, case):
    for seq_n in sequences:
        for band in bands:
            log.info(f"Processing sequence {seq_n} of band {band}")

            fname = (
                VT_SIM_DIR
                / f"{case}/qsrclist_{case}_{seq_n}/{sim_id}_QSCRLIST_{seq_n}_{band}.dat"
            )
            if not fname.exists():
                log.warning(
                    f"Sequence {band}{seq_n[-1]} missing for simulation {sim_id} of {case}"
                )
                continue

            tab = Table.read(
                fname,
                format="ascii",
                delimiter=r"\s",
                guess=False,
                fast_reader=False,
                names=[
                    "id",
                    "x",
                    "y",
                    "ra",
                    "dec",
                    "mag",
                    "magerr",
                    "flag",
                    "ellipticity",
                ],
            )
            tab.rename_columns(
                (
                    "id",
                    "x",
                    "y",
                    "ra",
                    "dec",
                    "mag",
                    "magerr",
                    "flag",
                    "ellipticity",
                ),
                (
                    "OBJID",
                    "X",
                    "Y",
                    "RA",
                    "DEC",
                    "MAGCAL",
                    "MAGERR",
                    "EFLAG",
                    "ELLIPTY",
                ),
            )
            hdr = read_header(fname)
            seq_name = "{}{}".format(hdr["BAND"][0].upper(), hdr["NTH"])
            hdr["SIM_ID"] = fname.stem.split("_")[0]
            hdr["SEQ"] = seq_name
            seq = seq_name

            # Add missing keywords
            hdr["MAGLIM"] = 23.0

            ext_name = f"FINDCH_{seq}"
            log.debug(f"Filling template's '{ext_name}' extension")
            # Create dynamical extension
            qsrclist_vt.add_dynamical_hdu("FINDCH_*", suffix=seq)

            # Initialize number of rows in table
            qsrclist_vt.init_bintable(len(tab), ext=ext_name)

            bin_tab = qsrclist_vt._fits_product[ext_name]

            # Fill binary table column by column
            for col_name, col_data in tab.columns.items():
                log.debug(f"Processing column {col_name}")

                if col_name not in bin_tab.columns.names:
                    log.warning(
                        f"Ignoring column '{col_name}' as it does not exist "
                        f"in FITS model for extension {ext_name}"
                    )
                    continue

                qsrclist_vt.fill_bintable_col(col_name, col_data, ext=ext_name)

            # Fill headers
            # Remove some keywords
            for k in [
                "SIM_ID",
                "NPCKTS",
                "DPACKETTIMES",
                "SEQ",
                "VHFNOBSNUMBER",
                "MSSGTYPE",
                "VHFIOBSTYPE",
            ]:
                hdr.pop(k)
            fill_headers(fits_template=qsrclist_vt, keywords_dict={ext_name: hdr})

    # Save reformatted QSRCLIST_VT
    qsrclist_fname = VT_SIM_DIR / f"{case}/fits/qsrclist_vt/{sim_id}_qsrclist_vt.fits"
    qsrclist_vt.writeto(qsrclist_fname, overwrite=True, checksum=True)
    return qsrclist_fname


def process_qpo_vt(qsrclist_fname, sim_id, case):
    hand = QuickPositionHandler(
        burst_id=sim_id,
        mode="local",
        cat_list=["PS1DR1", "SDSS12", "LSDR10"],
        qsrclist_vt_fname=qsrclist_fname,
        tmp_dir=VT_SIM_DIR / f"{case}/tmp/{sim_id}/",
    )
    hand.launch_processing()
    hand.create_qpo_vt(
        mode="local",
        json_product_descriptor_path=SVOM_DIR/"json-product-descriptor/",
    )
    qpo_vt_fname = VT_SIM_DIR / f"{case}/fits/qpo_vt/{sim_id}_qpo_vt.fits"
    hand.qpo_vt.writeto(qpo_vt_fname, overwrite=True, checksum=True)


def process_simulation(args):
    sim_id = args.get("sim_id")
    case = args.get("case")

    # Define handlers for logging
    log_handler_debug = add_logging_handler(case, sim_id, level="debug")
    log_handler_info = add_logging_handler(case, sim_id, level="info")
    t1 = time.time()
    log.info(f"Processing simulation {sim_id} for {case}")
    # log.warning(f"Logging handlers : {log.handlers}")

    # Define files
    qpo_vt_fname = VT_SIM_DIR / f"{case}/fits/qpo_vt/{sim_id}_qpo_vt.fits"
    qsrclist_fname = VT_SIM_DIR / f"{case}/fits/qsrclist_vt/{sim_id}_qsrclist_vt.fits"

    # Start timing
    t1 = time.time()

    try:
        if not qsrclist_fname.exists():
            # Get the fits template
            qsrclist_vt = get_product_template(
                acronym="QSRCLIST_VT",
                mode="local",
                json_product_descriptor_path="/Users/palmerio/SVOM_pipeline/json-product-descriptor/",
            )
            qsrclist_fname = reformat_qsrclist(qsrclist_vt, sim_id=sim_id, case=case)
        else:
            log.info(
                f"QSRCLIST_VT already exists for simulation {sim_id} of {case}"
            )

        if not qpo_vt_fname.exists():
            process_qpo_vt(qsrclist_fname, sim_id=sim_id, case=case)
        else:
            log.info(f"QPO_VT already exists for simulation {sim_id} of {case}")

    except Exception as exc:
        log.exception(
            f"Failed to process simulation {sim_id} because '{exc}'.\nMoving on to next simulation"
        )

    # End timing
    t2 = time.time()
    log.info(f"Processing time: {timedelta(seconds=(t2-t1))} ")

    # Remove handler for this sim_id
    remove_logging_handler(log_handler_debug)
    remove_logging_handler(log_handler_info)


if __name__ == "__main__":
    config_root_logger()
    # logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger()

    # Silence jpg logger as it is too verbose
    logging.getLogger("json_product_generator.fits_builder").setLevel(logging.WARNING)

    cli_args = get_cli_args()
    # Get the command line arguments
    case = cli_args.case

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

    if case not in cases:
        raise ValueError(f"Case must be one of {cases}")

    # for case in cases:
    # Create directories
    # To store fits products
    fits_dir = VT_SIM_DIR / f"{case}/fits"
    fits_dir.mkdir(exist_ok=True)

    # To store logs
    log_dir = VT_SIM_DIR / f"{case}/logs"
    # Clean logs from any previous run
    log.info("Cleaning old logs...")
    if log_dir.exists():
        for f in log_dir.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                for _f in f.iterdir():
                    _f.unlink()

    log_dir.mkdir(exist_ok=True)

    # for temporary files
    tmp_dir = VT_SIM_DIR / f"{case}/tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Simulation 71 is missing from bright_case1a, dunno why

    # args is a list where each element is a dictionary
    # containing the arguments to pass to process_simulation
    args = []
    # Get the simulation IDs
    sim_ids = [f"{i:07}" for i in np.arange(1, 365)]

    # for case in cases:
    _args = [{"case": case, "sim_id": sim_id} for sim_id in sim_ids]
    args += _args

    # Without multiprocessing
    for arg in args[:1]:
        process_simulation(arg)

    # BEWARE MULTIPROCESSING KILLS LOGGING AND REDEFINES THEIR OWN LEVELS
    # Prepare multiprocessing
    # n_cpu = multiprocessing.cpu_count()
    # # n_cpu = 1
    # n_sim = len(args)
    # log.info(f"Using {n_cpu} cpu")

    # with Pool(n_cpu) as p:
    #     for i, _ in enumerate(p.imap_unordered(process_simulation, args[:2]), 1):
    #         sys.stderr.write('\rdone {0:%}'.format(i/n_sim))
