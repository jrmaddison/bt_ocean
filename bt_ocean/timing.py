"""Timing utilities.
"""

import time


__all__ = \
    [
        "Timer"
    ]


class Timer:
    """Timing using :func:`time.perf_counter`.

    The timer starts on instantiation, and can be restarted using the
    :meth:`restart` method.
    """

    def __init__(self):
        self._restart()

    def _restart(self):
        self._t0 = time.perf_counter()

    def restart(self):
        """Restart the timer.

        Returns
        -------

        float
            The current timer time.
        """

        t = self.time()
        self._restart()
        return t

    def time(self):
        """Return the current timer time.

        Returns
        -------

        float
            The current timer time.
        """

        return time.perf_counter() - self._t0
