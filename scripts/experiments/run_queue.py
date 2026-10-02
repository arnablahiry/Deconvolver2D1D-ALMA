#!/usr/bin/env python
"""
Runs the experiment tasks on all GPUs: one task per GPU at a time, a task starts when
its dependencies have finished and its input files exist, longest tasks first. A GPU is
used only once its "ready" marker appears (a line in a log file), so this can wait for
earlier queues. Logs: logs/experiments/<task>.log; live state: logs/experiments/status.json.

    run_queue.py --gpus 0 1 2 3 --ready "0:<log>:<text>" ...
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/scratch/alahiry/conda/envs/denoise2d1d/bin/python"
RUN = os.path.join(REPO, "scripts/experiments/run_model.py")
ED, EF = "data/experiments", "figures/experiments"
LOGS = os.path.join(REPO, "logs/experiments")


def tasks():
    T = []
    def add(name, args, minutes, deps=(), needs=(), script=RUN):
        T.append(dict(name=name, cmd=[PY, "-u", script] + args, minutes=minutes, deps=list(deps), needs=list(needs)))
    # ---- 4: primal-dual with FISTA's reweighting reference (tau * lambda), with GIFs
    for cfg, ks, mins in (("compact", (2, 3, 4, 5), 12), ("extended", (3, 4, 5), 55)):
        for k in ks:
            add(f"04_pdtau_{cfg}_k{k}", ["--method", "pdtau", "--cfg", cfg, "--k", str(k),
                "--out", f"{ED}/04_pd_tau_reweighting/{cfg}/pdtau_k{k}.npy",
                "--gif", f"{EF}/04_pd_tau_reweighting/{k}_sigma/{cfg}"], mins)
    # ---- 1: simulations with known truth
    for m in ("pd", "pdtau", "fista"):
        for k in (2, 3, 4, 5):
            add(f"01_sim_compact_{m}_k{k}", ["--method", m, "--cfg", "compact", "--sim", "--k", str(k),
                "--out", f"{ED}/01_simulations/compact/{m}_k{k}.npy"], 10)
        add(f"01_sim_extended_{m}_k3", ["--method", m, "--cfg", "extended", "--sim", "--k", "3",
            "--out", f"{ED}/01_simulations/extended/{m}_k3.npy"], 50)
    # ---- 3: debiasing (5000 iterations from each model)
    base = {("compact", "pd", 2): "data/ngc3110/compact/pd/model.npy",
            ("compact", "fista", 2): "data/ngc3110/compact/fista/model_k2_20k.npy"}
    for k in (3, 4, 5):
        for m in ("pd", "fista"):
            base[("compact", m, k)] = f"data/ngc3110/compact/{m}/model_k{k}_20k.npy"
            base[("extended", m, k)] = f"data/ngc3110/extended/{m}/model_k{k}_20k.npy"
    for (cfg, m, k), x0 in base.items():
        add(f"03_debias_real_{cfg}_{m}_k{k}", ["--method", "debias", "--cfg", cfg, "--k", str(k), "--n-iter", "5000",
            "--x0", x0, "--out", f"{ED}/03_debiasing/real/{cfg}/{m}_k{k}_debiased.npy"], 3 if cfg == "compact" else 14, needs=[x0])
    for cfg, ks in (("compact", (2, 3, 4, 5)), ("extended", (3, 4, 5))):
        for k in ks:
            x0 = f"{ED}/04_pd_tau_reweighting/{cfg}/pdtau_k{k}.npy"
            add(f"03_debias_real_{cfg}_pdtau_k{k}", ["--method", "debias", "--cfg", cfg, "--k", str(k), "--n-iter", "5000",
                "--x0", x0, "--out", f"{ED}/03_debiasing/real/{cfg}/pdtau_k{k}_debiased.npy"],
                3 if cfg == "compact" else 14, deps=[f"04_pdtau_{cfg}_k{k}"])
    for m in ("pd", "pdtau", "fista"):
        for cfg, ks in (("compact", (2, 3, 4, 5)), ("extended", (3,))):
            for k in ks:
                x0 = f"{ED}/01_simulations/{cfg}/{m}_k{k}.npy"
                add(f"03_debias_sim_{cfg}_{m}_k{k}", ["--method", "debias", "--cfg", cfg, "--sim", "--k", str(k), "--n-iter", "5000",
                    "--x0", x0, "--out", f"{ED}/03_debiasing/sim/{cfg}/{m}_k{k}_debiased.npy"],
                    3 if cfg == "compact" else 14, deps=[f"01_sim_{cfg}_{m}_k{k}"])
    # ---- 5: compact on a 32" field (0.04")
    for k in (2, 3):
        add(f"05_large_field_pd_k{k}", ["--method", "pd", "--cfg", "compact", "--grid", "large", "--k", str(k),
            "--out", f"{ED}/05_large_field/pd_k{k}.npy"], 45, needs=["data/ngc3110/compact/cube_32arcsec/dirty_beam_cube.fits"])
    # ---- 7: NGC 2369 blind test with the unchanged recipe
    mask = "data/ngc2369/extended/pd/support_mask.npy"
    add("07_ngc2369_grow_mask", ["--galaxy", "ngc2369", "--out", mask], 15, script=os.path.join(REPO, "scripts/experiments/grow_mask.py"))
    add("07_ngc2369_extended_pd_k3", ["--method", "pd", "--galaxy", "ngc2369", "--cfg", "extended", "--k", "3", "--mask", mask,
        "--out", f"{ED}/07_ngc2369/extended_pd_k3.npy"], 55, deps=["07_ngc2369_grow_mask"])
    add("07_ngc2369_compact_pd_k2", ["--method", "pd", "--galaxy", "ngc2369", "--cfg", "compact", "--k", "2",
        "--out", f"{ED}/07_ngc2369/compact_pd_k2.npy"], 12)
    # ---- 8: error maps, parametric bootstrap of the final compact recipe
    for i in range(10):
        add(f"08_bootstrap_{i}", ["--method", "pd", "--cfg", "compact", "--k", "2", "--bootstrap", str(i),
            "--bootstrap-model", "data/ngc3110/compact/pd/model.npy", "--out", f"{ED}/08_error_maps/bootstrap_{i}.npy"], 12)
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--ready", nargs="*", default=[]); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    ready = {}
    for r in a.ready:
        g, path, text = r.split(":", 2); ready[g] = (path, text)
    T = tasks()
    if a.only:
        T = [t for t in T if t["name"].startswith(a.only)]
    os.makedirs(LOGS, exist_ok=True); os.chdir(REPO)
    out_of = lambda t: t["cmd"][t["cmd"].index("--out") + 1] if "--out" in t["cmd"] else None
    state = {t["name"]: ("done" if out_of(t) and os.path.exists(out_of(t)) else "pending") for t in T}
    for n, st in state.items():
        if st == "done":
            print(f"{n}: output exists, skipped", flush=True)
    stuck = 0
    running = {}                                             # gpu -> (task, proc, t0)
    order = sorted(T, key=lambda t: -t["minutes"])
    def gpu_ready(g):
        if g not in ready:
            return True
        p, text = ready[g]
        return os.path.exists(p) and text in open(p).read()
    while any(s in ("pending", "running") for s in state.values()):
        for g, (t, proc, t0) in list(running.items()):
            if proc.poll() is not None:
                state[t["name"]] = "done" if proc.returncode == 0 else "failed"
                print(f"{time.strftime('%H:%M')} {t['name']}: {state[t['name']]} ({(time.time() - t0) / 60:.0f} min, GPU {g})", flush=True)
                del running[g]
        for t in order:
            if state[t["name"]] == "pending" and any(state.get(d) == "failed" for d in t["deps"]):
                state[t["name"]] = "failed"; print(f"{t['name']}: skipped (a dependency failed)", flush=True)
        for g in a.gpus:
            if g in running or not gpu_ready(g):
                continue
            for t in order:
                if (state[t["name"]] == "pending" and all(state.get(d, "done") == "done" for d in t["deps"])
                        and all(os.path.exists(n) for n in t["needs"])):
                    log = open(os.path.join(LOGS, t["name"] + ".log"), "w")
                    proc = subprocess.Popen(t["cmd"], stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=g))
                    running[g] = (t, proc, time.time()); state[t["name"]] = "running"
                    print(f"{time.strftime('%H:%M')} start {t['name']} on GPU {g}", flush=True)
                    break
        if not running and any(gpu_ready(g) for g in a.gpus) and "pending" in state.values():
            stuck += 1
            if stuck > 30:                                   # ~10 min with nothing startable: inputs will not appear
                for t in order:
                    if state[t["name"]] == "pending":
                        state[t["name"]] = "failed"; print(f"{t['name']}: failed (inputs never appeared: {t['needs']})", flush=True)
        else:
            stuck = 0
        json.dump(dict(state=state, running={g: r[0]["name"] for g, r in running.items()}),
                  open(os.path.join(LOGS, "status.json"), "w"), indent=1)
        time.sleep(20)
    n_fail = sum(s == "failed" for s in state.values())
    print(f"ALL EXPERIMENT TASKS FINISHED: {len(state) - n_fail} done, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
