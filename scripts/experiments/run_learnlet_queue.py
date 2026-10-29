#!/usr/bin/env python
"""
Experiment 22 queue: starlet vs learnlet 2D dictionaries (x CDF 9/7), NGC 3110 compact, pdtau, 20,000 iterations.

  null   pure-noise data (measured power spectrum, seed 7): every dictionary at k = 3, 4, 5, 6, and the learnlet
         threshold profile -> learnlet_calibrate.py: the k with the starlet's k = 5 spurious flux
  real   the learnlet threshold profile; the calibrated k (the k = 5 runs are in data/experiments/22_learnlet/)
  sim    compact ground-truth simulation: k = 5, the threshold profile, the calibrated k

One task per GPU at a time once its dependencies are done: the calibration's chain (nulls, calibrate, calibrated runs)
first, longest first within each group; a task whose output exists is skipped
(rerun-safe). A run starts on a GPU only if no other user's process is on it (this user's own jobs are shared
with) and its free memory covers the run's peak (NEED_GB). Logs: logs/22_learnlet/<task>.log; status: logs/22_learnlet/status.md (Europe/Athens times).

    nohup python scripts/experiments/run_learnlet_queue.py --gpus 1 2 3 &
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/scratch/alahiry/conda/envs/deconvolver2d1d/bin/python"
RUN = os.path.join(REPO, "scripts/experiments/run_model.py")
ED = "data/experiments/22_learnlet"
LOGS = os.path.join(REPO, "logs/22_learnlet")
CAL = f"{ED}/calibration.json"
ALL_DICTS = {"starlet": ("starlet", 12), "learnlet16": ("learnlet:16", 105), "learnlet": ("learnlet", 195),
             "learnletfire": ("learnlet_fire", 105)}   # minutes; learnletfire: scripts/train_learnlet_fire.py
DICTS = dict(ALL_DICTS)                                       # narrowed by --dicts
now = lambda: datetime.now(ZoneInfo("Europe/Athens")).strftime("%Y-%m-%d %H:%M")


def tasks():
    T = []

    def add(name, tag, args, out, deps=(), needs=(), gpu=True, minutes=None):
        T.append(dict(name=name, env=dict(DICT_2D=DICTS[tag][0]) if tag else {},
                      minutes=minutes if minutes is not None else (DICTS[tag][1] if tag else 0),
                      need_gb=NEED_GB.get(tag, 0),
                      cmd=[PY, "-u", RUN, "--method", "pdtau", "--galaxy", "ngc3110", "--cfg", "compact"] + args
                      + ["--out", out] if gpu else args, out=out, deps=list(deps), needs=list(needs), gpu=gpu))

    for tag in DICTS:
        for k in (3, 4, 5, 6):
            add(f"null_{tag}_k{k}", tag, ["--k", str(k), "--null", "7"], f"{ED}/null/{tag}_k{k}.npy")
    learnt = [t for t in DICTS if t != "starlet"]
    for tag in DICTS:                                         # real data at k = 5 where experiment 22's first runs lack it
        top = f"{ED}/ngc3110_compact_pdtau_k5_{tag}.npy"
        add(f"real_{tag}_k5", tag, ["--k", "5"], top)
    for tag in learnt:
        add(f"null_{tag}_profile", tag, ["--k", "5", "--null", "7", "--lam-profile", "learnlet"], f"{ED}/null/{tag}_profile.npy")
        add(f"real_{tag}_profile", tag, ["--k", "5", "--lam-profile", "learnlet"], f"{ED}/real/{tag}_profile.npy")
        add(f"sim_{tag}_profile", tag, ["--k", "5", "--sim", "--lam-profile", "learnlet"], f"{ED}/sim/{tag}_profile.npy")
    for tag in DICTS:
        add(f"sim_{tag}_k5", tag, ["--k", "5", "--sim"], f"{ED}/sim/{tag}_k5.npy")
    fire = "learnletfire" in DICTS
    LOWK = (1, 1.5, 2)                                        # the FIRE prior is far more conservative at equal k
    if fire:
        for k in LOWK:
            add(f"null_learnletfire_k{k:g}", "learnletfire", ["--k", f"{k:g}", "--null", "7"], f"{ED}/null/learnletfire_k{k:g}.npy")
    add("calibrate", None, [PY, os.path.join(REPO, "scripts/experiments/learnlet_calibrate.py")], CAL,
        deps=[f"null_{t}_k{k}" for t in DICTS for k in (3, 4, 5, 6)] + ([f"null_learnletfire_k{k:g}" for k in LOWK] if fire else []),
        gpu=False)
    for tag in learnt:
        for kind, extra in (("real", []), ("sim", ["--sim"])):
            add(f"{kind}_{tag}_kcal", tag, ["--k-from", f"{CAL}:{tag}"] + extra, f"{ED}/{kind}/{tag}_kcal.npy",
                deps=["calibrate"], needs=[CAL])
    if fire:                                                  # the starlet's trade-off curve over the same range
        for k in (1, 1.5, 6):
            add(f"sim_starlet_k{k:g}", "starlet", ["--k", f"{k:g}", "--sim"], f"{ED}/sim/starlet_k{k:g}.npy")
        for k in (1, 1.5, 2):
            add(f"null_starlet_k{k:g}", "starlet", ["--k", f"{k:g}", "--null", "7"], f"{ED}/null/starlet_k{k:g}.npy")
    if fire:                                                  # (1) trade-off curve, (2) convergence, (3) rebalanced prior
        t = "learnletfire"
        for k in (1, 1.5, 2, 3):
            add(f"sim_{t}_k{k:g}", t, ["--k", f"{k:g}", "--sim"], f"{ED}/sim/{t}_k{k:g}.npy")
        add(f"sim_{t}_k5_it60k", t, ["--k", "5", "--sim", "--n-iter", "60000"], f"{ED}/sim/{t}_k5_it60k.npy",
            minutes=3 * DICTS[t][1])
        add(f"sim_{t}_k5_warm", t, ["--k", "5", "--sim", "--x0", f"{ED}/sim/starlet_k5.npy"], f"{ED}/sim/{t}_k5_warm.npy",
            needs=[f"{ED}/sim/starlet_k5.npy"])
        for w in (0.25, 0.1):
            add(f"sim_{t}_w{w:g}_k5", t, ["--k", "5", "--sim", "--learnt-weight", f"{w:g}"], f"{ED}/sim/{t}_w{w:g}_k5.npy")
            add(f"null_{t}_w{w:g}_k5", t, ["--k", "5", "--null", "7", "--learnt-weight", f"{w:g}"], f"{ED}/null/{t}_w{w:g}_k5.npy")
    return T


def external(out, mine):
    """a run_model process writing out that this queue did not start (e.g. left by a previous queue process)"""
    r = subprocess.run(["pgrep", "-f", "--", f"--out {out}"], capture_output=True, text=True).stdout.split()
    return any(int(p) not in mine for p in r if p.strip())


quick = lambda t: t["name"].endswith("_k5") and t["name"].startswith(("real_", "sim_"))   # the direct comparison first
NEED_GB = {"starlet": 6, "learnlet16": 18, "learnlet": 33, "learnletfire": 18}           # peak GPU memory per run, with margin


def gpu_ok(g, mine, need_gb):
    """GPU g can take a run needing need_gb: no other user's process on it (this user's own jobs are shared with),
    and enough free memory"""
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    uuid = q("-i", str(g), "--query-gpu=uuid").strip()
    me = os.environ.get("USER") or subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()
    for line in q("--query-compute-apps=gpu_uuid,pid").splitlines():
        if not line.strip():
            continue
        u, pid = [x.strip() for x in line.split(",")]
        if u == uuid and int(pid) not in mine:
            owner = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
            if owner and owner != me:
                return False
            cmd = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout
            if "run_model.py" in cmd:                         # a solve not started by this queue process
                return False
    return float(q("-i", str(g), "--query-gpu=memory.free").strip()) / 1024 >= need_gb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", nargs="+", required=True)
    ap.add_argument("--dicts", nargs="+", default=list(ALL_DICTS), choices=list(ALL_DICTS),
                    help="2D dictionaries to run (starlet is the reference and always included)")
    a = ap.parse_args()
    for t in list(DICTS):
        if t != "starlet" and t not in a.dicts:
            del DICTS[t]
    os.makedirs(LOGS, exist_ok=True)
    T = tasks()
    global CRITICAL                                           # the calibration's chain runs first, then the rest
    cal = next(t for t in T if t["name"] == "calibrate")
    CRITICAL = set(cal["deps"]) | {"calibrate"} | {t["name"] for t in T if "calibrate" in t["deps"]}
    state = {t["name"]: ("done" if os.path.exists(os.path.join(REPO, t["out"])) else
                         "external" if t["gpu"] and external(t["out"], set()) else "pending") for t in T}
    times, running = {}, {}                                   # running: gpu -> (task, Popen)
    free = list(a.gpus)
    while True:
        for t in T:                                           # runs left by a previous queue process
            if state[t["name"]] == "external":
                if os.path.exists(os.path.join(REPO, t["out"])):
                    state[t["name"]] = "done"
                elif not external(t["out"], {p.pid for _, p in running.values()}):
                    state[t["name"]] = "pending"
        for g, (t, p) in list(running.items()):
            if p.poll() is not None:
                ok = p.returncode == 0 and os.path.exists(os.path.join(REPO, t["out"]))
                state[t["name"]] = "done" if ok else f"FAILED ({p.returncode})"
                times[t["name"]] = (times[t["name"]][0], now())
                del running[g]
                if g != "cpu":
                    free.append(g)
        ready = sorted([t for t in T if state[t["name"]] == "pending" and all(state[d] == "done" for d in t["deps"])
                        and all(os.path.exists(os.path.join(REPO, n)) for n in t["needs"])],
                       key=lambda t: (not quick(t), t["name"] not in CRITICAL, "_profile" in t["name"], -t["minutes"]))
        for t in ready:
            if t["gpu"]:
                mine = {p.pid for _, p in running.values()}
                g = next((g for g in free if gpu_ok(g, mine, t["need_gb"])), None)
                if g is None:
                    continue
                free.remove(g)
            else:
                g = "cpu"
            env = dict(os.environ, PYTHONUNBUFFERED="1", **t["env"])
            if t["gpu"]:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            os.makedirs(os.path.dirname(os.path.join(REPO, t["out"])), exist_ok=True)
            p = subprocess.Popen(t["cmd"], cwd=REPO, env=env, stdout=open(os.path.join(LOGS, t["name"] + ".log"), "w"),
                                 stderr=subprocess.STDOUT)
            running[g] = (t, p); state[t["name"]] = f"running (GPU {g})" if t["gpu"] else "running"
            times[t["name"]] = (now(), "")
        with open(os.path.join(LOGS, "status.md"), "w") as f:
            f.write(f"# Experiment 22 queue (updated {now()} Athens)\n\n| task | 2D | status | started | finished |\n|---|---|---|---|---|\n")
            for t in T:
                st, fi = times.get(t["name"], ("", ""))
                f.write(f"| {t['name']} | {t['env'].get('DICT_2D', '')} | {state[t['name']]} | {st} | {fi} |\n")
        if not running and not ready and "external" not in state.values():
            blocked = [n for n, s in state.items() if s == "pending"]
            print(f"{now()} queue finished" + (f"; blocked by failures: {blocked}" if blocked else ""), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
