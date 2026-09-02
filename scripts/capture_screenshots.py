"""Capture the README screenshots from the running application.

Requires Playwright (``pip install playwright && playwright install chromium``)
and both servers running:

    uvicorn app.main:app --app-dir backend
    cd frontend && npm run dev

Then:

    python scripts/capture_screenshots.py
    python scripts/capture_screenshots.py --base-url http://localhost:5173 --dark

Writes PNGs into ``docs/screenshots/``. If Playwright is not installed the
script says so and exits rather than failing obscurely - screenshots are a
documentation nicety, not part of the pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "screenshots"

# (filename, path, viewport, description)
SHOTS = [
    ("home.png", "/", (1440, 960), "Landing page and uploader"),
    ("dashboard.png", "/dashboard", (1440, 1200), "Prediction dashboard"),
    ("benchmark.png", "/benchmark", (1440, 1400), "Model benchmark"),
    ("research-dataset.png", "/research", (1440, 1300), "Research - dataset"),
    ("research-ablation.png", "/research?tab=ablation", (1440, 1700),
     "Research - attention ablation and CBAM placement"),
    ("research-calibration.png", "/research?tab=calibration", (1440, 1400),
     "Research - calibration and out-of-distribution analysis"),
    ("library.png", "/library", (1440, 1100), "Disease library"),
    ("history.png", "/history", (1440, 1000), "Prediction history"),
    ("mobile.png", "/", (390, 900), "Mobile layout"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:5173")
    parser.add_argument("--dark", action="store_true", help="Capture the dark theme.")
    parser.add_argument("--wait", type=int, default=2500,
                        help="Milliseconds to wait after load for charts to render.")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. To capture screenshots:\n"
            "  pip install playwright\n"
            "  playwright install chromium\n"
            "Screenshots are documentation only - the pipeline does not need them."
        )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    theme = "dark" if args.dark else "light"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for filename, route, (width, height), description in SHOTS:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=2,
                is_mobile=width < 768,
            )
            page = context.new_page()
            # Set the theme before the app boots so it does not flash and then swap.
            page.add_init_script(f"localStorage.setItem('pdd-theme', '{theme}')")
            page.goto(f"{args.base_url}{route}", wait_until="networkidle")
            page.wait_for_timeout(args.wait)

            destination = OUTPUT_DIR / filename
            page.screenshot(path=str(destination), full_page=height > 1000)
            print(f"  {destination.relative_to(PROJECT_ROOT)}  ({description})")
            context.close()
        browser.close()

    print(f"\nWrote {len(SHOTS)} screenshots to docs/screenshots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
