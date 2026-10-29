#!/usr/bin/env python
"""
Run scripts/pipeline/run_target.py for each target in turn (one CASA process at a time), each into
results/<target>/pipeline.log; a failed target is recorded and the queue moves on. Status table (Europe/Athens
times) in results/QUEUE.md, rewritten after every target.

    nohup /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/pipeline/run_queue.py [targets ...] &
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
# single-pointing targets first; the two mosaics (central pointing only) last
DEFAULT = ["NGC2369", "ESO297-G011", "NGC1614", "IC4518E", "IC4518W", "IRASF17138-1017", "NGC7130", "IC5179",
           "NGC5135", "IC4687", "MCG-02-33-098", "NGC3256"]
now = lambda: datetime.now(ZoneInfo("Europe/Athens")).strftime("%Y-%m-%d %H:%M")


def write(rows):
    with open(os.path.join(RESULTS, "QUEUE.md"), "w") as fh:
        fh.write("# Pipeline queue (scripts/pipeline/run_queue.py)\n\nTimes: Europe/Athens. Per-target log: "
                 "results/<target>/pipeline.log; steps done: results/<target>/.done/.\n\n"
                 "| target | status | started | finished | hours |\n|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['t']} | {r['status']} | {r.get('start', '')} | {r.get('end', '')} | {r.get('hours', '')} |\n")


def main():
    targets = sys.argv[1:] or DEFAULT
    rows = [dict(t=t, status="queued") for t in targets]
    write(rows)
    for r in rows:
        os.makedirs(os.path.join(RESULTS, r["t"]), exist_ok=True)
        r.update(status="running", start=now()); write(rows)
        t0 = time.time()
        with open(os.path.join(RESULTS, r["t"], "pipeline.log"), "a") as fh:
            rc = subprocess.run([sys.executable, os.path.join(REPO, "scripts/pipeline/run_target.py"), r["t"]],
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        r.update(status="done" if rc == 0 else f"FAILED (exit {rc}, see pipeline.log)", end=now(),
                 hours=f"{(time.time() - t0) / 3600:.2f}")
        write(rows)


if __name__ == "__main__":
    main()
