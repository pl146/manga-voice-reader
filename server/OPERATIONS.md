# Manga Voice Reader — Operations Baseline

## PC Startup Flow
1. **On login**: Scheduled task `MangaLauncher` runs `pythonw C:\manga_server\launcher.py`
2. **Launcher** (port 5056): Listens for `/start` requests, launches server with `CREATE_NEW_CONSOLE`
3. **Server** (port 5055): Loads models (~13s), serves Flask on `0.0.0.0:5055`
4. **Extension**: Tries PC server first → launcher → Mac fallback

## Thread Cap Defaults (config.py)
| Env Var | Default | Controls |
|---------|---------|----------|
| `MVR_CPU_THREAD_CAP` | 2 | OMP/MKL/BLAS/Paddle threads |
| `MVR_ONNX_THREAD_CAP` | 2 | ONNX Runtime inter/intra threads |
| `MVR_TORCH_THREAD_CAP` | 2 | PyTorch CPU threads |

Set to `0` to disable caps.

## Verification Checklist
```
curl http://192.168.2.183:5055/health     # expect {"status":"ok","florence":true}
curl http://192.168.2.183:5056/status     # expect {"running":true}
```
Then: open manga page → click read → confirm bubbles detected + TTS plays.

## Rollback
```powershell
# Remove thread caps (restore library defaults):
set MVR_CPU_THREAD_CAP=0
set MVR_ONNX_THREAD_CAP=0
set MVR_TORCH_THREAD_CAP=0
# Then restart server

# Remove scheduled task:
schtasks /delete /tn "MangaLauncher" /f
```

## Expected Behavior
- **GPU fans ramp briefly** during /process requests — RTX 3080 doing Florence-2 inference. Normal. Duration scales with bubble count (~300ms/bubble).
- **System CPU peaks ~23-26%** during detection burst (0.8s), settles to ~16% during OCR phase.
- **Server uses ~3.5GB RAM** with all models loaded.
- **Auto-shutdown** after 30 min idle (configurable via `MVR_IDLE_TIMEOUT`).

## Benchmark
Disabled by default. To enable: `set MVR_ENABLE_BENCHMARK=true`, restart server, then POST to `/benchmark` from localhost only.
