import logging
from logging.config import dictConfig
import my_module
from vtac.catalogs import get_catalogs

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, filename=log_file)

    # create logger
    dictConfig(
        {
            "version": 1,
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
            },
            "loggers": {
                "": {
                    "handlers": [
                        "console",
                    ],
                    "level": "DEBUG",
                    "propagate": True,
                },

            },
        }
    )

    log = logging.getLogger()

    log.info("Starting ! (this should be only written to console)")

    n = 3

    for i in range(n):

        log.debug(f"Running iteration {i+1} of {n} ({100*i/n:.1f}%) (only console)")

        # Add a file handler for this execution
        log_handler = logging.FileHandler(filename=f"{i}.log")
        log_handler.setLevel(logging.DEBUG)

        log.addHandler(log_handler)

        log.info("About to launch my_func (console and logfile)")

        # Run some code
        my_module.my_func(i)
        get_catalogs()

        # Remove file handler
        log.removeHandler(log_handler)

    log.info("Finishing script (this should be only written to console)")