import sys
import platform

mods = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("matplotlib", "mpl"),
    ("scipy", "sp"),
    ("sklearn", "sk"),
    ("numba", "numba"),
    ("plotly", "plotly"),
    ("yaml", "yaml"),
]

print("Python:", sys.version)
print("Platform:", platform.platform())
print("OK: interpreter up.")

for m, alias in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "n/a")
        print(f"OK: {m} {ver}")
    except Exception as e:
        print(f"FAIL: {m} -> {type(e).__name__}: {e}")
        raise SystemExit(1)

print("Smoke test passed.")
