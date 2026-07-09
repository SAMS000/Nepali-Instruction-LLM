import importlib
import sys

import torch


def check_package(package_name: str):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "Unknown")
        print(f"✓ {package_name:<20} {version}")
    except ImportError:
        print(f"✗ {package_name:<20} Not Installed")


print("=" * 60)
print("Environment Check")
print("=" * 60)

print(f"Python: {sys.version.split()[0]}")
print()

packages = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
]

for package in packages:
    check_package(package)

print()

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print(f"GPU Count: {torch.cuda.device_count()}")

print("=" * 60)