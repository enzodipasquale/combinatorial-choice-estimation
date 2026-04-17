import os, time
from pathlib import Path

job_name = os.environ.get("JOB_NAME", "local")
out_dir = Path("test_automation/results")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{job_name}.txt"
out.write_text(f"hello from {job_name} at {time.ctime()}\n")
print(f"wrote {out}")