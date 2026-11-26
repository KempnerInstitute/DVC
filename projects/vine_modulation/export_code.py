"""Export the study and its reusable estimators from one committed DVC checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PROJECT = "projects/vine_modulation"
PACKAGE = "src/dvc_package/modulation"


def package(repo, revision, output):
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo), *args])
    pin = git("rev-parse", revision + "^{commit}").decode().strip()
    listing = git("show", f"{pin}:{PROJECT}/release_files.txt").decode().splitlines()
    if len(listing) != len(set(listing)) or listing != sorted(listing):
        raise ValueError("The release file list must be unique and sorted.")
    for path in listing:
        if path in ("LICENSE", "src/dvc_package/__init__.py"):
            continue
        if not path.startswith((PROJECT + "/", PACKAGE + "/")) or ".." in Path(path).parts:
            raise ValueError(f"Unexpected source path: {path}")
    source = {name: name for name in listing}
    source["README.md"] = PROJECT + "/README.md"
    if PROJECT + "/experiments/e31_challenge.py" not in source or PACKAGE + "/vinemodel2.py" not in source:
        raise ValueError("The revision must contain the DVC study and its library implementation.")
    allowed = {".py", ".md", ".txt", ".R", ".tsv", ".csv", ".json"}
    manifest = {}
    with ZipFile(output, "w") as archive:
        def write(name, data):
            entry = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compress_type=ZIP_DEFLATED)
        for name, path in sorted(source.items()):
            if path != "LICENSE" and Path(path).suffix not in allowed:
                raise ValueError(f"Unexpected study artifact: {path}")
            data = git("show", f"{pin}:{path}")
            manifest[name] = {"source": path, "sha256": hashlib.sha256(data).hexdigest()}
            write(name, data)
        doc = {"repository": "KempnerInstitute/DVC", "source_commit": pin, "project": PROJECT, "files": manifest}
        write("MANIFEST.json", (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode())
    print(f"{len(source)} source files, {output.stat().st_size:,} bytes, DVC {pin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dvc_checkout", type=Path)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(args.dvc_checkout, args.revision, args.output)
