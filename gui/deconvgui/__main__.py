"""
Start the GUI server.

    deconvgui                              # laptop: opens a browser tab
    deconvgui --port 8765 --root /nas/datasets/ALMA_visibilities --workdir /scratch/$USER/deconvgui

On a cluster, the server listens on 127.0.0.1 only; reach it through an SSH
tunnel (the exact command is printed at start-up).
"""

import argparse
import os
import secrets
import socket
import sys
import webbrowser


def main(argv=None):
    p = argparse.ArgumentParser(prog="deconvgui", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to listen on (default 127.0.0.1: only via localhost / SSH tunnel)")
    p.add_argument("--workdir", default=os.environ.get("DECONVGUI_WORKDIR", "~/deconvgui_work"),
                   help="where datasets and results are stored (default ~/deconvgui_work)")
    p.add_argument("--root", action="append", default=None,
                   help="extra directory the file browser may open (repeatable; home and CWD are always allowed; the first --root is where the browser starts)")
    p.add_argument("--token", default=os.environ.get("DECONVGUI_TOKEN"),
                   help="access token (default: $DECONVGUI_TOKEN, else a random one kept in <workdir>/.token "
                        "so the URL stays the same across restarts)")
    p.add_argument("--no-token", action="store_true", help="disable the access token (single-user laptop only)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = p.parse_args(argv)

    import uvicorn
    from .server import create_app

    # --root adds to home and CWD; the browser opens at the first root, so given ones go first
    roots = list(dict.fromkeys(os.path.realpath(os.path.expanduser(r))
                               for r in (args.root or []) + ["~", os.getcwd()]))
    token = None if args.no_token else (args.token or _saved_token(args.workdir))

    # Fail before printing a URL: if another deconvgui still holds the port,
    # its token differs and the printed link would only give "bad token".
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((args.host, args.port))
        except OSError:
            sys.exit(f"deconvgui: port {args.port} on {args.host} is already in use "
                     f"(another deconvgui still running?). Stop it or pass --port.")

    app = create_app(args.workdir, roots, token)

    host = socket.gethostname()
    url = f"http://localhost:{args.port}/" + (f"?token={token}" if token else "")
    user = os.environ.get("USER", "you")
    print("\n  deconvgui is running on", host)
    print(f"  workdir : {os.path.abspath(os.path.expanduser(args.workdir))}")
    print(f"  open    : {url}")
    print(f"  via SSH : ssh -N -L {args.port}:{host}:{args.port} {user}@<login-node>   (then open the URL above)\n")
    sys.stdout.flush()
    # Over plain SSH there is no local browser; VS Code Remote sets $BROWSER to a
    # helper that opens the tab on the laptop and forwards the port.
    if not args.no_browser and (os.environ.get("BROWSER") or not os.environ.get("SSH_CONNECTION")):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _saved_token(workdir):
    """Random token, generated once and kept in <workdir>/.token (mode 600)."""
    workdir = os.path.abspath(os.path.expanduser(workdir))
    path = os.path.join(workdir, ".token")
    try:
        with open(path) as f:
            token = f.read().strip()
        if token:
            return token
    except FileNotFoundError:
        pass
    os.makedirs(workdir, exist_ok=True)
    token = secrets.token_urlsafe(16)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token


if __name__ == "__main__":
    main()
