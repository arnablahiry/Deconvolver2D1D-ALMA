#!/usr/bin/env python
"""
Shared no-jupyter/nbformat-needed harness for this repo's `build_*_notebook.py`
scripts: each of those only needs to declare its own sequence of markdown/code
cells via `NotebookBuilder.md`/`.code`, then call `.build_and_execute()`.

The harness executes each code cell in one shared namespace (so later cells
see earlier cells' variables, exactly like running the notebook top-to-bottom
in Jupyter), captures stdout and any matplotlib figures as cell outputs, and
writes the result as a standard .ipynb file. It stops at the first cell that
raises, embeds the traceback as an error output, and exits non-zero -- so a
build failure is loud rather than silently producing a truncated notebook.
"""

import base64
import io
import os
import sys
import json
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")


class NotebookBuilder:
    def __init__(self, out_path, chdir_to_notebooks=False):
        """out_path: where to write the .ipynb (relative to REPO_ROOT or absolute).
        chdir_to_notebooks: run cells with cwd = notebooks/, matching a real
        Jupyter session started there -- needed when cell sources use paths
        like '../data' relative to that directory."""
        self.out_path = out_path if os.path.isabs(out_path) else os.path.join(REPO_ROOT, out_path)
        self.chdir_to_notebooks = chdir_to_notebooks
        self.cells = []

    def md(self, text):
        self.cells.append(("markdown", text))

    def code(self, text):
        self.cells.append(("code", text))

    def build_and_execute(self):
        sys.path.insert(0, SRC_DIR)
        if self.chdir_to_notebooks:
            os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
            os.chdir(NOTEBOOKS_DIR)

        namespace = {}
        nb_cells = []

        for i, (ctype, source) in enumerate(self.cells):
            source = source.strip("\n")
            if ctype == "markdown":
                nb_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": source.splitlines(keepends=True),
                })
                continue

            print(f"--- executing code cell {i} ---", file=sys.stderr)
            stdout_buf = io.StringIO()
            outputs = []
            error = None
            plt.close("all")
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(compile(source, f"<cell {i}>", "exec"), namespace)
            except Exception as exc:  # noqa: BLE001
                error = exc
            finally:
                text = stdout_buf.getvalue()
                if text:
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": text.splitlines(keepends=True),
                    })
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("ascii")
                    outputs.append({
                        "output_type": "display_data",
                        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                        "metadata": {},
                    })
                plt.close("all")

            if error is not None:
                outputs.append({
                    "output_type": "error",
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "traceback": [f"{type(error).__name__}: {error}"],
                })

            nb_cells.append({
                "cell_type": "code",
                "execution_count": i,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                "outputs": outputs,
            })

            if error is not None:
                print(f"!!! cell {i} raised {error!r}", file=sys.stderr)
                break

        notebook = {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": sys.version.split()[0]},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(notebook, f, indent=1)
        print(f"Wrote {self.out_path}")

        had_error = any(
            any(o.get("output_type") == "error" for o in c.get("outputs", []))
            for c in nb_cells if c["cell_type"] == "code"
        )
        if had_error:
            print("NOTEBOOK BUILD HAD AN ERROR -- see traceback above", file=sys.stderr)
            sys.exit(1)
