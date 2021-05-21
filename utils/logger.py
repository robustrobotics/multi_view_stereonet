# Copyright 2021 Massachusetts Institute of Technology
#
# @file logger.py
# @author W. Nicholas Greene
# @date 2019-12-08 17:39:52 (Sun)

import logging

def create_logger(fname, level=logging.INFO):
    """Create a log file.

    Based on: https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create outputs.
    if not logger.handlers:
        # Log to file.
        file_handler = logging.FileHandler(fname)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Log to screen.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    return logger
