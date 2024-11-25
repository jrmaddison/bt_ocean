def _init():
    import keras

    import logging
    import sys
    import warnings

    if keras.backend.backend() != "jax":
        warnings.warn("bt_ocean requires Keras with the JAX backend", ImportWarning)

    logger = logging.getLogger("bt_ocean")

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_init()
del _init
