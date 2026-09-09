$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Prepend the venv's own Scripts dir to PATH (what Activate.ps1 does).
# Without this, a bare "python"/"python.exe" lookup by name -- which
# CUDA torch's own driver-safety probe subprocess does internally on
# import, separately from anything this project or uvicorn controls
# -- resolves via PATH to the global Python install instead of this
# venv, silently losing CUDA torch there.
$env:PATH = "$scriptDir\venv\Scripts;$env:PATH"

# NOT --reload: uvicorn's reload mode spawns its actual worker via
# multiprocessing, which on this machine re-execs using the global
# Python install instead of this venv (a Windows venv/multiprocessing
# quirk) -- silently losing CUDA torch and falling back to the
# global install's broken/CPU-only torch. Restart this script by hand
# after code changes instead.
& "$scriptDir\venv\Scripts\python.exe" -m uvicorn backend.main:app --port 8000
