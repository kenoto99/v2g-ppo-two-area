"""Write SHA-256 checksums of the final outputs (results, figures, tables, manuscript PDFs)
and the exact package versions to results/CHECKSUMS.sha256 and results/ENVIRONMENT.txt."""
import glob, hashlib, os, platform, subprocess, sys
ROOT = os.path.join(os.path.dirname(__file__), ".."); os.chdir(ROOT)
MS = sys.argv[1] if len(sys.argv) > 1 else "../manuscript"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


files = sorted(glob.glob("results/eval/*.json") + glob.glob("results/eval_modes/*/*.json") + glob.glob("results/ppo_*/*.json")
               + glob.glob("results/ppo_*/best_model.zip") + glob.glob("results/ppo_*/final_model.zip")
               + glob.glob("results/tex/*") + glob.glob("figures/*.pdf") + glob.glob("results/*.json") + glob.glob("results/*.md")
               + glob.glob("v2g/*.py") + glob.glob("scripts/*.py") + glob.glob("scripts/*.sh") + glob.glob("tests/*.py")
               + [p for p in [f"{MS}/access.pdf", f"{MS}/response_to_reviewers.pdf", f"{MS}/supplementary.pdf", f"{MS}/diff/access_diff.pdf"] if os.path.exists(p)])
with open("results/CHECKSUMS.sha256", "w") as f:
    for p in files:
        f.write(f"{sha(p)}  {os.path.relpath(p, ROOT)}\n")
import numpy, scipy, torch, gymnasium, stable_baselines3, pandapower, numba, matplotlib
with open("results/ENVIRONMENT.txt", "w") as f:
    f.write(f"python {platform.python_version()} ({platform.platform()})\n")
    for m in [numpy, scipy, torch, gymnasium, stable_baselines3, pandapower, numba, matplotlib]:
        f.write(f"{m.__name__}=={m.__version__}\n")
    f.write(f"cpu threads {os.cpu_count()}\n")
    f.write(subprocess.run(["pdflatex", "--version"], capture_output=True, text=True).stdout.splitlines()[0] + "\n")
print(f"{len(files)} files hashed")
