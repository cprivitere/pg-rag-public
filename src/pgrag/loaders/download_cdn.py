import os
import sys

import requests

from pgrag.config import CDN_DIR

sys.stdout.reconfigure(line_buffering=True)

CDN_BASE = "https://cdn.projectgorgon.com"
VERSION_URL = "http://client.projectgorgon.com/fileversion.txt"

DATA_FILES = [
    "items",
    "skills",
    "abilities",
    "recipes",
    "effects",
    "npcs",
    "areas",
    "attributes",
    "xptables",
    "advancementtables",
    "abilitykeywords",
    "abilitydynamicdots",
    "abilitydynamicspecialvalues",
    "ai",
    "directedgoals",
    "itemuses",
    "landmarks",
    "lorebooks",
    "lorebookinfo",
    "playertitles",
    "quests",
    "sources_abilities",
    "sources_items",
    "sources_recipes",
    "storagevaults",
    "tsysclientinfo",
    "tsysprofiles",
]


def get_remote_version():
    resp = requests.get(VERSION_URL, timeout=10)
    resp.raise_for_status()
    return int(resp.text.strip())


def get_cached_version():
    path = CDN_DIR / "version.txt"
    if path.exists():
        return int(path.read_text().strip())
    return None


def download_cdn():
    remote_ver = get_remote_version()
    local_ver = get_cached_version()

    if local_ver == remote_ver:
        print(f"CDN data is up-to-date (v{remote_ver})")
        return True

    print(f"New version detected: v{local_ver} -> v{remote_ver}")
    CDN_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATA_FILES:
        url = f"{CDN_BASE}/v{remote_ver}/data/{name}.json"
        dest = CDN_DIR / f"{name}.json"
        print(f"  Downloading {name}.json...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        # Temp + os.replace so a crash mid-download never leaves a partial file.
        temp = dest.with_suffix(".json.tmp")
        temp.write_bytes(resp.content)
        os.replace(temp, dest)

    temp_ver = (CDN_DIR / "version.txt").with_suffix(".txt.tmp")
    temp_ver.write_text(str(remote_ver))
    os.replace(temp_ver, CDN_DIR / "version.txt")
    print(f"Done. Saved to {CDN_DIR}")
    return True


if __name__ == "__main__":
    download_cdn()
