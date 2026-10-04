"""Colab VM, once per session: clone main at 2db3f51 and download PIQA."""
import os
import subprocess


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{(r.stdout + r.stderr).strip()[-600:]}", flush=True)
    r.check_returncode()


sh("rm -rf /content/rob && git clone -q -b main https://github.com/SEKY443/Group69TinyLanguageModel.git /content/rob")
sh("git -C /content/rob checkout -q 2db3f51 && git -C /content/rob log --oneline -1")
sh("pip -q install gdown")
if not os.path.exists("/content/piqa/train.jsonl"):
    sh("mkdir -p /content/piqa && cd /content/piqa && gdown -q --folder "
       "https://drive.google.com/drive/folders/1lxEsHLbRsgOHh8rOAd75QHUZds7ybKtT -O . && "
       "find . -type f -name '*.*' -path '*/*/*' -exec mv {} . \\; ; ls")
print("SETUP OK")
