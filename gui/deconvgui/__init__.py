"""deconvgui: a local browser front end for Deconvolver2D1D-ALMA.

The server runs where the data are (laptop or cluster node); the browser only
receives small arrays, PNG/GIF frames and statistics. Every job runs in a
fresh worker process that imports the repository's `simple/` modules at that
moment, so edits to the solver are picked up by the next job without
restarting the app.
"""

__version__ = "0.1.0"
