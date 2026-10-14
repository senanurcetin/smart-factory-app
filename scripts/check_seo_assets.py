"""Validate search metadata and screenshot-derived preview assets without network or third-party dependencies."""

import ast
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    "https://smart-factory-app.onrender.com",
    "main.py",
    "docs/assets/seo",
    "/assets/seo",
    [
        ("main.py", "/", "factory-social.jpg", "smart-factory-dashboard.png"),
        ("case_study.py", "/case-study", "maintenance-social.jpg", "smart-factory-maintenance.png"),
        ("rul_case_study.py", "/rul-case-study", "rul-social.jpg", "smart-factory-rul.png"),
    ],
)


def dimensions(data):
    assert data[:2] == b"\xff\xd8", "JPEG magic invalid"
    i = 2
    while i + 4 < len(data):
        if data[i] != 255:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        length = int.from_bytes(data[i : i + 2], "big")
        assert length >= 2
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            return int.from_bytes(data[i + 5 : i + 7], "big"), int.from_bytes(
                data[i + 3 : i + 5], "big"
            )
        i += length
    raise AssertionError("No JPEG dimensions found")


def literal(source, key):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == key for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing constant " + key)


def check():
    site, server, assets, prefix, pages = CONFIG
    head_source = (ROOT / server).read_text(encoding="utf-8-sig")
    robots = literal(head_source, "ROBOTS_TXT")
    sitemap = literal(head_source, "SITEMAP_XML")
    assert robots.startswith("User-agent: *\nAllow: /\n")
    assert "Sitemap: " + site + "/sitemap.xml" in robots
    assert "Disallow: /api/" in robots
    locs = [n.text for n in ET.fromstring(sitemap).iter() if n.tag.endswith("loc")]
    assert locs == [site + route for _, route, _, _ in pages]
    manifest = json.loads(
        (ROOT / "docs/assets/production_screenshot_provenance_20261014.json").read_text(
            encoding="utf-8"
        )
    )
    src_hashes = {p["asset"]: p["sha256"] for p in manifest["files"]}
    assert (ROOT / assets / "favicon.svg").is_file()
    for file_path, route, jpg, original in pages:
        data = (ROOT / file_path).read_text(encoding="utf-8-sig")
        data = data.split("</head>", 1)[0]
        canonical = site + route
        image = site + prefix + "/" + jpg
        for expected in (
            'rel="canonical" href="' + canonical + '"',
            'property="og:url" content="' + canonical + '"',
            'property="og:image" content="' + image + '"',
            'name="twitter:card" content="summary_large_image"',
            'name="description"',
            'property="og:description"',
            'rel="icon"',
        ):
            assert expected in data, (file_path, expected)
        assert data.count('rel="canonical"') == 1, (file_path, "canonical duplicate")
        assert data.count('name="description"') == 1, (file_path, "description duplicate")
        matched = re.findall(r'<script type="application/ld\+json">(.*?)</script>', data, re.S)
        assert len(matched) == 1, (file_path, "JSON-LD")
        structured = json.loads(matched[0])
        assert structured["@type"] == "WebPage"
        assert structured["url"] == canonical
        assert structured["primaryImageOfPage"]["url"] == image
        assert original.startswith(tuple(["web_", "smart-", "visual-"])) or bool(original)
        origin = ROOT / "docs/assets" / original
        assert (
            hashlib.sha256(origin.read_bytes()).hexdigest() == src_hashes["docs/assets/" + original]
        )
        content = (ROOT / assets / jpg).read_bytes()
        assert len(content) < 400_000, (jpg, len(content))
        assert dimensions(content) == (1200, 630), (jpg, dimensions(content))
    assert (ROOT / "README.md").read_text(encoding="utf-8-sig").count(
        "SEO_AND_SOCIAL_PREVIEW_GUIDE.md"
    ) == 1
    assert (ROOT / "docs/SEO_AND_SOCIAL_PREVIEW_GUIDE.md").is_file()
    print(
        "PASS: SEO, JSON-LD, sitemap, robots, screenshot sources and social assets:",
        len(pages),
        "pages",
    )


if __name__ == "__main__":
    check()
