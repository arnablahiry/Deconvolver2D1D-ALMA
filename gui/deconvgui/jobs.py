"""
Job manager: one worker process per job, a reader thread per job that turns
the worker's messages into an in-memory state the browser polls.

Device scheduling: a job may name a list of acceptable devices
(e.g. ["cuda:0", "cuda:1"] or ["mps"] or ["cpu"]). Each device runs one job at
a time; a job waits in the queue until one of its devices is free, then runs
on that device. Submitting several runs (e.g. a sweep over k) with several
devices selected therefore spreads them over the devices.
"""

import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections import deque

from .store import REPO_ROOT
from . import worker


class Job:
    def __init__(self, kind, label, ds_id=None, res_id=None, devices=None):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.label = label
        self.ds_id = ds_id
        self.res_id = res_id
        self.devices = list(devices) if devices else None
        self.device = None
        self.status = "queued" if devices else "running"   # queued | running | done | error | cancelled
        self.fraction = None
        self.message = "waiting for a free device: " + ", ".join(devices) if devices else "starting worker"
        self.log = deque(maxlen=400)
        self.stats = []                 # per-iteration diagnostics (deconvolve)
        self.frame = None               # latest moment-0 frame (float32 2D)
        self.frame_iter = -1
        self.frame_seq = 0
        self.result = None
        self.error = None
        self.created = time.time()
        self.started = None
        self.finished = None
        self.cancel = None
        self.proc = None
        self.args = None
        self.on_done = None
        self.on_start = None

    def summary(self, stats_since=0):
        return dict(id=self.id, kind=self.kind, label=self.label, ds_id=self.ds_id, res_id=self.res_id,
                    devices=self.devices, device=self.device,
                    status=self.status, fraction=self.fraction, message=self.message,
                    log=list(self.log)[-200:], stats=self.stats[stats_since:], n_stats=len(self.stats),
                    frame_iter=self.frame_iter, frame_seq=self.frame_seq, result=self.result,
                    error=self.error, created=self.created, started=self.started, finished=self.finished)


class JobManager:
    def __init__(self, store):
        self.store = store
        self.jobs = {}
        self.pending = []               # queued jobs, FIFO
        self.busy = {}                  # device -> job id
        self.ctx = mp.get_context("spawn")
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    def start(self, kind, label, args, ds_id=None, res_id=None, on_done=None, on_start=None, devices=None):
        """devices=None: run now (inspection, imaging). devices=[...]: queue
        until one of them is free; the chosen one is passed as args['device']."""
        job = Job(kind, label, ds_id, res_id, devices)
        job.args, job.on_done, job.on_start = dict(args), on_done, on_start
        with self.lock:
            self.jobs[job.id] = job
            if devices:
                self.pending.append(job)
                self._schedule()
            else:
                self._launch(job, None)
        return job

    def _schedule(self):
        with self.lock:
            for job in list(self.pending):
                free = [d for d in job.devices if d not in self.busy]
                if not free:
                    continue
                self.pending.remove(job)
                self._launch(job, free[0])

    def _launch(self, job, device):
        if device is not None:
            self.busy[device] = job.id
            job.device = device
            job.args["device"] = device
        job.status, job.started, job.message = "running", time.time(), f"starting worker on {device or 'cpu'}"
        q = self.ctx.Queue()
        job.cancel = self.ctx.Event()
        job.proc = self.ctx.Process(target=worker.run_job, args=(job.kind, job.args, q, job.cancel, REPO_ROOT),
                                    daemon=True)
        job.proc.start()
        if job.on_start:
            try:
                job.on_start(job)
            except Exception as e:
                job.log.append(f"[gui] on_start failed: {e!r}")
        threading.Thread(target=self._pump, args=(job, q), daemon=True).start()

    # ------------------------------------------------------------------
    def _pump(self, job, q):
        while True:
            try:
                kind, payload = q.get(timeout=0.5)
            except queue.Empty:
                if not job.proc.is_alive():
                    try:
                        kind, payload = q.get(timeout=0.5)
                    except queue.Empty:
                        if job.status == "running":
                            if job.cancel.is_set():
                                job.status, job.message = "cancelled", "cancelled"
                            else:
                                job.status = "error"
                                job.error = f"worker exited unexpectedly (exit code {job.proc.exitcode})"
                            job.finished = time.time()
                            self._finish(job)
                        return
                else:
                    continue
            if kind == "log":
                job.log.append(payload)
            elif kind == "progress":
                job.fraction = payload.get("fraction")
                job.message = payload.get("message", job.message)
            elif kind == "stat":
                job.stats.append(payload)
            elif kind == "frame":
                job.frame_iter, job.frame = payload
                job.frame_seq += 1
            elif kind in ("result", "error", "cancelled"):
                job.status = {"result": "done", "error": "error", "cancelled": "cancelled"}[kind]
                if kind == "result":
                    job.result, job.fraction, job.message = payload, 1.0, "done"
                elif kind == "error":
                    job.error = payload
                    job.log.append(payload)
                    job.message = "failed"
                else:
                    job.message = "cancelled"
                job.finished = time.time()
                self._finish(job)
                job.proc.join(timeout=5)
                return

    def _finish(self, job):
        with self.lock:
            if job.device is not None and self.busy.get(job.device) == job.id:
                del self.busy[job.device]
        if job.on_done:
            try:
                job.on_done(job)
            except Exception as e:      # never let bookkeeping kill the pump
                job.log.append(f"[gui] post-processing failed: {e!r}")
        self._schedule()

    # ------------------------------------------------------------------
    def get(self, job_id):
        return self.jobs[job_id]

    def cancel(self, job_id):
        job = self.jobs[job_id]
        with self.lock:
            if job.status == "queued":
                self.pending.remove(job)
                job.status, job.message, job.finished = "cancelled", "cancelled before it started", time.time()
                if job.on_done:
                    job.on_done(job)
                return job
        if job.status == "running":
            job.cancel.set()
            # the solvers check the flag every iteration; CASA does not, so
            # terminate the process if it has not stopped after a grace period
            def _kill():
                time.sleep(8)
                if job.proc.is_alive():
                    job.proc.terminate()
            threading.Thread(target=_kill, daemon=True).start()
        return job

    def list(self):
        return [j.summary(stats_since=10 ** 9) for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def device_load(self):
        with self.lock:
            return dict(busy=dict(self.busy), queued=[(j.id, j.devices) for j in self.pending])
