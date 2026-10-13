"""Serialize access to the shared, process-local Smart Factory demo state.

Flask may serve concurrent requests in one worker. The simulation mutates
counters, event logs and a rolling telemetry deque; CSV export must not read
that deque while a scoring request is updating it.

This lock only coordinates threads in the same process. Separate Gunicorn
workers have independent simulated state, as documented in the dashboard.
"""

from functools import wraps
from threading import RLock


_demo_state_lock = RLock()


def synchronized_demo_view(view):
    """Run a Flask view atomically with respect to other demo-state views."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        with _demo_state_lock:
            return view(*args, **kwargs)

    wrapped.demo_state_synchronized = True
    return wrapped


def advance_production_counts(actual, scrap, completed_units, scrap_event):
    """Advance simulated plant counts without scrapping an unproduced unit.

    The demo treats each API tick as a batch of 0–3 completed units. At most
    one member of that batch can be scrapped. The invariant is maintained:
    0 <= scrap <= actual.
    """
    if (
        not isinstance(completed_units, int)
        or isinstance(completed_units, bool)
        or completed_units < 0
        or not 0 <= scrap <= actual
    ):
        raise ValueError("invalid simulated production counters")
    actual += completed_units
    if completed_units > 0 and scrap_event:
        scrap += 1
    return actual, scrap
