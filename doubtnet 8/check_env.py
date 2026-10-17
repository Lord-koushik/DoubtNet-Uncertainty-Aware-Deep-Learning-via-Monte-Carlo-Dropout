"""Run this FIRST:  python3 check_env.py
Tells you exactly which step is broken and how to fix it."""
import sys
import time

HINTS = {
    "torch": "pip3 install torch torchvision",
    "torchvision": "pip3 install torchvision",
    "sklearn": "pip3 install scikit-learn",
    "matplotlib": "pip3 install matplotlib",
    "certifi": "pip3 install certifi",
    "download": ("If the error says CERTIFICATE_VERIFY_FAILED (common on macOS python.org Python):\n"
                 "      open Finder -> Applications -> 'Python 3.x' folder -> double-click "
                 "'Install Certificates.command', then re-run.\n"
                 "      Otherwise check your internet / VPN / college proxy."),
}


def step(name, fn, hint_key=None):
    t0 = time.time()
    try:
        out = fn()
        print(f"  OK   {name}" + (f"  ({out})" if out else "") + f"  [{time.time() - t0:.1f}s]", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        if hint_key:
            print(f"    -> fix: {HINTS[hint_key]}")
        return False


def main():
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    ok = True
    for mod in ["numpy", "sklearn", "matplotlib", "PIL", "certifi"]:
        ok &= step(f"import {mod}", lambda m=mod: __import__(m).__name__ and "", mod if mod in HINTS else None)
    print("  (importing torch for the first time can take 30-90 s on a Mac - this is normal)")
    ok &= step("import torch", lambda: __import__("torch").__version__, "torch")
    ok &= step("import torchvision", lambda: __import__("torchvision").__version__, "torchvision")
    if not ok:
        print("\nFix the FAIL lines above, then run this again.")
        sys.exit(1)

    import torch
    from doubtnet.model import DoubtNet, get_device

    ok &= step("device", lambda: str(get_device()))
    ok &= step("model forward pass", lambda: f"out {tuple(DoubtNet()(torch.randn(2, 1, 28, 28)).shape)}")

    def dl():
        from doubtnet import data
        data.mnist_test("data")
        return "MNIST ready"
    ok &= step("download MNIST", dl, "download")

    def dl2():
        from doubtnet import data
        data.ood_datasets("data", n=10)
        return "Fashion-MNIST + KMNIST ready"
    ok &= step("download Fashion-MNIST / KMNIST", dl2, "download")

    print("\nALL GOOD - now run:  python3 run_all.py" if ok else
          "\nSome steps failed - fix them (hints above) and re-run, or send me this output.")


if __name__ == "__main__":
    main()
