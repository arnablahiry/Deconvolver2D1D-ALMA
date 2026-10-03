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
                   help="directory the file browser may open (repeatable; default: home and CWD)")
    p.add_argument("--token", default=None, help="access token (default: random)")
    p.add_argument("--no-token", action="store_true", help="disable the access token (single-user laptop only)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = p.parse_args(argv)

    import uvicorn
    from .server import create_app

    roots = args.root or [os.path.expanduser("~"), os.getcwd()]
    token = None if args.no_token else (args.token or secrets.token_urlsafe(16))
    app = create_app(args.workdir, roots, token)

    host = socket.gethostname()
    url = f"http://localhost:{args.port}/" + (f"?token={token}" if token else "")
    user = os.environ.get("USER", "you")
    print("\n  deconvgui is running on", host)
    print(f"  workdir : {os.path.abspath(os.path.expanduser(args.workdir))}")
    print(f"  open    : {url}")
    print(f"  via SSH : ssh -N -L {args.port}:{host}:{args.port} {user}@<login-node>   (then open the URL above)\n")
    sys.stdout.flush()
    if not args.no_browser and not os.environ.get("SSH_CONNECTION"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
