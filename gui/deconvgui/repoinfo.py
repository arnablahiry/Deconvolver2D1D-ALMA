"""
Read-only view of the repository version shown in the header.

Nothing here writes to the repository: no fetch, no pull, no commit.
`GIT_OPTIONAL_LOCKS=0` stops `git status` from refreshing the index (which
would otherwise create .git/index.lock).
"""

import os
import subprocess

from .store import REPO_ROOT


def _git(*args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        r = subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True,
                           timeout=10, env=env)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def repo_info():
    head = _git("rev-parse", "--short", "HEAD")
    if head is None:
        return dict(git=False, root=REPO_ROOT)
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    lines = [l for l in status.splitlines() if l.strip()]
    conflicts = [l[3:] for l in lines if l[:2] in ("UU", "AA", "DD", "AU", "UA", "DU", "UD")]
    return dict(
        git=True, root=REPO_ROOT,
        commit=head,
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        subject=_git("log", "-1", "--format=%s"),
        date=_git("log", "-1", "--format=%cd", "--date=short"),
        modified=len(lines),
        conflicts=conflicts,
        # last known remote state (whatever YOUR last fetch/pull recorded)
        behind=_int(_git("rev-list", "--count", "HEAD..@{u}")),
        ahead=_int(_git("rev-list", "--count", "@{u}..HEAD")),
    )


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None
