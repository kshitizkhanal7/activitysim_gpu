"""Run ActivitySim with the experimental combined-direction logsum path."""

from __future__ import annotations

import sys


def main() -> int:
    from activitysim.abm.models import trip_destination
    from activitysim.cli import main as activitysim_main
    from choiceforge.activitysim_destination import compute_logsums_combined

    original = trip_destination.compute_logsums

    def combined(*args, **kwargs):
        return compute_logsums_combined(*args, **kwargs, fallback=original)

    trip_destination.compute_logsums = combined
    try:
        try:
            return int(activitysim_main.main() or 0)
        except SystemExit as exc:
            return int(exc.code or 0)
    finally:
        trip_destination.compute_logsums = original


if __name__ == "__main__":
    raise SystemExit(main())
