"""Download the 4 experiment images from Wikimedia Commons by exact file title.

Records provenance (page URL, author, license) into images/sources.json.
"""

import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
UA = "CLIP-homework/1.0 (educational use; contact: student) Python-urllib"

# (Commons file title, local filename)
FILES = [
    ("File:Happy 2.jpg", "01_dog.jpg"),
    ("File:Cat playing with a lizard.jpg", "02_cat.jpg"),
    ("File:Red Apple.jpg", "03_apple.jpg"),
    ("File:Sopwith Triplane replica flying overhead in blue sky.jpg", "04_airplane.jpg"),
]


def api(params):
    params = dict(params, format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def clean(html):
    """Strip HTML markup that the Commons API embeds in metadata fields."""
    text = re.sub(r"<[^>]+>", " ", str(html))
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return " ".join(text.split())


records = []
for title, filename in FILES:
    data = api({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1200",
    })
    page = next(iter(data["query"]["pages"].values()))
    info = page["imageinfo"][0]
    dest = OUT / filename
    req = urllib.request.Request(info.get("thumburl") or info["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    meta = info.get("extmetadata", {})
    records.append({
        "file": filename,
        "commons_title": title,
        "description_page": info.get("descriptionurl"),
        "download_url": (info.get("thumburl") or info["url"]).split("?")[0],
        "original_size": f"{info['width']}x{info['height']}",
        "artist": clean(meta.get("Artist", {}).get("value", "")).strip(" ."),
        "credit": clean(meta.get("Credit", {}).get("value", "")),
        "license": clean(meta.get("LicenseShortName", {}).get("value", "")),
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "image_description": clean(meta.get("ImageDescription", {}).get("value", "")),
        "date": clean(meta.get("DateTimeOriginal", {}).get("value", "")),
    })
    print(f"saved {filename}: {title} ({info['mime']}, {info['width']}x{info['height']})")
    time.sleep(2)

with open(OUT / "sources.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print("wrote", OUT / "sources.json")

ATTRIBUTION = ["# Image sources and licences", "",
               "All four photos come from Wikimedia Commons and are used under the licence",
               "stated for each file. Downloaded at 1280 px width via the Commons API.",
               ""]
for rec in records:
    license_text = rec["license"] + (f" ({rec['license_url']})" if rec["license_url"] else "")
    ATTRIBUTION += [
        f"## `{rec['file']}`",
        "",
        f"- File page: {rec['description_page']}",
        f"- Author: {rec['artist']}",
        f"- Licence: {license_text}",
        f"- Original size: {rec['original_size']}",
        f"- Description: {rec['image_description']}",
        f"- Direct file: {rec['download_url']}",
        "",
    ]
(OUT / "ATTRIBUTION.md").write_text("\n".join(ATTRIBUTION), encoding="utf-8")
print("wrote", OUT / "ATTRIBUTION.md")
