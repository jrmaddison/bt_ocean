def _init():
    import logging
    import sys

    logger = logging.getLogger("bt_ocean")

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_init()
del _init
