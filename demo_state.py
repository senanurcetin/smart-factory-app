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
