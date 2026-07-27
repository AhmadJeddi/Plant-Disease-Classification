"""
Export Installed Package Versions
---------------------------------
Utility script to export the exact
versions of core dependencies used in the
Plant Disease Classification project.

This helps:
- Reproducibility
- Environment tracking
- Creating requirements.txt
- Sharing stable experiment setups

Output:
requirements.txt
"""

from importlib.metadata import version

# =====================================
# PACKAGE LIST
# =====================================
PACKAGE_NAMES = [
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "scikit-learn",
    "matplotlib",
]

# =====================================
# EXPORT REQUIREMENTS FILE
# =====================================
OUTPUT_FILE = "requirements.txt"

with open(OUTPUT_FILE, "w") as f:

    for package in PACKAGE_NAMES:

        try:
            package_version = version(package)

            line = f"{package}=={package_version}"

            print(line)

            f.write(line + "\n")

        except Exception as e:

            print(f"⚠️ Could not get version for {package}: {e}")

print(f"\n✅ Requirements exported to: {OUTPUT_FILE}")
