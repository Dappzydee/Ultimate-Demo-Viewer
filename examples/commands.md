# Example commands

Install dependencies first with `python -m pip install -r requirements.txt`.

```powershell
python vision.py match.dem --player "donk" --round 1 --start 1:00 --end 1:20 --out seen_result.glb
python flash-events.py match.dem --json flashes.json
python flash-coverage.py --flash-json flashes.json --flash-index 12 --out flash_12.glb
python viewer.py out/seen_result.glb
```

These are examples only. The README and manual are the user-facing documentation for supported options and assumptions.
