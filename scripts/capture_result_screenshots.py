"""Capture the diagnosis result page for the README.

``capture_screenshots.py`` photographs every page in its empty state. The most
informative screen - a diagnosis with its heat maps - only exists after an
upload, so this script drives a real one: it opens the Analyse page, puts an
image in the file input, clicks Analyse and photographs the result.

    python scripts/capture_result_screenshots.py --base-url http://127.0.0.1:8000 \
        --image path/to/leaf.jpg --out docs/screenshots/result-diagnosis.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--tab", default=None,
                        help="Explanation tab to open before capturing, e.g. 'Grad-CAM++'.")
    parser.add_argument("--height", type=int, default=0,
                        help="Clip the capture to this many CSS pixels from the result's top "
                             "(0 = the whole result).")
    args = parser.parse_args()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        theme = "dark" if args.dark else "light"
        # Set before first paint, as capture_screenshots.py does, so there is no
        # flash of the other theme in the capture.
        page.add_init_script(f"try {{ localStorage.setItem('theme', '{theme}'); }} catch (e) {{}}")
        page.goto(args.base_url, wait_until="networkidle")
        page.set_input_files("input[type=file]", str(args.image))
        page.get_by_role("button", name="Analyse this leaf").click()
        result = page.locator(".animate-fade-in-up").first
        result.wait_for(state="visible", timeout=120_000)
        page.wait_for_timeout(1500)
        if args.tab:
            tab = page.get_by_role("button", name=args.tab)
            if tab.count():
                tab.first.click()
                page.wait_for_timeout(800)
        # The sticky navigation bar would be drawn over the top of an element
        # capture, and the "disease detected" toast over the heat-map card.
        page.evaluate("""() => {
            for (const el of document.querySelectorAll('body *')) {
                const position = getComputedStyle(el).position;
                if (position === 'sticky') el.style.position = 'static';
                if (position === 'fixed') el.style.display = 'none';
            }
        }""")
        page.wait_for_timeout(300)
        result.scroll_into_view_if_needed()
        box = result.bounding_box()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if box and args.height:
            page.screenshot(path=str(args.out), full_page=True,
                            clip={"x": max(0, box["x"] - 24), "y": max(0, box["y"] - 24),
                                  "width": box["width"] + 48, "height": args.height})
        else:
            result.screenshot(path=str(args.out))
        browser.close()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
