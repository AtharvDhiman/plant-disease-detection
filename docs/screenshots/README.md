# Screenshots

Captured from the running application, not mocked up.

## How to regenerate

Start both servers:

```bash
uvicorn app.main:app --app-dir backend
```

```bash
cd frontend && npm run dev
```

Then capture:

```bash
pip install playwright && playwright install chromium
python scripts/capture_screenshots.py
```

For the dark theme:

```bash
python scripts/capture_screenshots.py --dark
```

`scripts/capture_screenshots.py` sets the stored theme before the app boots (so
there is no flash of the wrong theme), waits for charts to finish rendering, and
captures at a 2× device pixel ratio. The mobile shot uses a 390 px viewport with
mobile emulation enabled.

## Expected files

| File | View |
|---|---|
| `home.png` | Landing page and uploader |
| `dashboard.png` | Prediction dashboard |
| `benchmark.png` | Model benchmark |
| `research-dataset.png` | Research dashboard, dataset tab |
| `library.png` | Disease library |
| `history.png` | Prediction history |
| `mobile.png` | Mobile layout |
