import subprocess
import sys
import time
import gc
import torch
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scripts = [
    # os.path.join(BASE, "CSQ_BLIP.py"),
    # os.path.join(BASE, "CSQ_BLIP_2.py"),
    # os.path.join(BASE, "CSQ_BLIP_3.py"),
    # os.path.join(BASE, "CSQ_BLIP_4.py"),
    os.path.join(BASE, "CSQ_BLIP_5.py"),
    # os.path.join(BASE, "CSQ_BLIP_6.py"),
    os.path.join(BASE, "CSQ_BLIP_7.py"),
]

for script in scripts:
    print(f"\n========== Start {script} ==========")
    start_time = time.time()
    result = subprocess.run([sys.executable, script])
    end_time = time.time()
    elapsed = end_time - start_time

    if result.returncode != 0:
        print(f"⚠️  {script} failed with exit code {result.returncode}")
        break

    # 防御性清理显存（一般不会触发，但更安全）
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    h, m = divmod(int(elapsed // 60), 60)
    s = int(elapsed % 60)
    print(f"✅  Finished {script} at {time.strftime('%H:%M:%S')}  ⏱️ 总耗时: {h:02d}h {m:02d}m {s:02d}s")
