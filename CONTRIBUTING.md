# Contributing

FlashRL is a reference RL data plane. Changes should preserve the token-level
contracts in `src/flashrl/schemas.py` and keep the distinction between a
portable reference implementation and a production GPU kernel explicit.

From the project directory:

```powershell
python -m pip install -e ".[torch,dev]"
python -m pytest -q
python -m compileall -q src tests scripts
python -m flashrl.cli ablation --samples 64 --output runs/ablation/local.json
```

Multi-process checks use `scripts/run_multigpu.ps1` on CUDA Linux/WSL or
`scripts/run_multigpu_local.py` for the Windows file-store smoke path. Keep
generated `runs/` artifacts out of commits. New public claims about
DeepSeek-V4.1-Flash should include a source link and a coverage note.
