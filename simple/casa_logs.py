#!/usr/bin/env python
"""
Keep CASA's own logging out of the repo root, in every script/notebook that
touches casatools.

Why this needs an atexit hook, not just `logsink().setlogfile()`
------------------------------------------------------------------
`casatools` writes a `casa-<timestamp>.log` into cwd the instant it is
imported -- before any tool exists to call `setlogfile` on. That first stray
file is easy to sweep up front. But casaconfig's data/measures "installed or
checked less than 1 day ago" startup check does not run at import time either
-- it runs lazily, the first time any actual tool (`msmetadata()`, `table()`,
...) is constructed, and writes through a logger of its own that ignores an
earlier `setlogfile` call entirely (verified empirically: calling
`setlogfile` right after `import casatools`, then constructing tools, still
drops a second stray `casa-*.log` into cwd on the *first* tool construction
only -- never on the second, third, ...). There is no earlier hook to catch
that one pre-emptively, so this sweeps again at process exit instead.
"""

import atexit
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASA_LOGS_DIR = os.path.join(REPO_ROOT, "casa_logs")


def _sweep_stray_logs():
    # cwd too: notebook cells (and their worker processes) run from notebooks/
    for d in {REPO_ROOT, os.getcwd()}:
        if not os.path.isdir(d):
            continue
        os.makedirs(CASA_LOGS_DIR, exist_ok=True)
        for f in os.listdir(d):
            if f.startswith("casa-") and f.endswith(".log"):
                try:
                    shutil.move(os.path.join(d, f), os.path.join(CASA_LOGS_DIR, f))
                except FileNotFoundError:
                    pass    # swept concurrently by another process (worker pools)


def redirect_casa_logs(log_name):
    """Call once, right after `import casatools`, before constructing any
    tool. `log_name` should identify the calling script/notebook, e.g.
    "inspect_ms.log"."""
    import casatools

    _sweep_stray_logs()
    path = os.path.join(CASA_LOGS_DIR, log_name)
    casatools.logsink().setlogfile(path)
    # casatasks (tclean, ...) log through their own `casalog` sink, which
    # the call above does not redirect.
    try:
        from casatasks import casalog
        casalog.setlogfile(path)
    except ImportError:
        pass
    atexit.register(_sweep_stray_logs)  # catches the lazy first-tool stray log
