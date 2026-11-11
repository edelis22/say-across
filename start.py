"""Set up and open a local copy of Click."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Preview the interface using fixed examples.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the local page.")
    parser.add_argument("--no-browser", action="store_true", help="Print the link without opening a browser.")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Use Python 3.11 to start Click. See START_HERE.md.")

    project = Path(__file__).resolve().parent
    environment = project / ".venv"
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not executable.is_file():
        print("Creating a local Python environment for Click…", flush=True)
        venv.create(environment, with_pip=True)

    install = [str(executable), "-m", "pip", "--disable-pip-version-check", "install", "--quiet"]
    print("Checking Click dependencies…", flush=True)
    if args.demo:
        subprocess.run([*install, "gradio==5.49.1"], cwd=project, check=True)
    else:
        if sys.platform in {"linux", "win32"}:
            # Match the container's CPU build and avoid installing CUDA libraries.
            subprocess.run([*install, "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.8.0"],
                           cwd=project, check=True)
        subprocess.run([*install, "-r", "requirements.txt"], cwd=project, check=True)
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            # Only needed for the optional restating and lookup features.
            subprocess.run([*install, "-r", "requirements-assistant.txt"], cwd=project, check=True)

    command = [str(executable), "-u", "app.py", "--port", str(args.port)]
    if args.demo:
        command.append("--demo")
    if not args.no_browser:
        command.append("--open-browser")
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(project / ".model-cache"))
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return subprocess.call(command, cwd=project, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Setup failed with exit code {error.returncode}. Check the installer message above.") from None
