# Environmental Monitoring

Python tools and notebooks for environmental monitoring with FLIR thermal imaging, GPS logging, HDC1080 temperature/humidity sensing, OLED status display, and Plotly-based data visualisation.

## Repository contents

```text
Environmental_Monitoring/
├── scripts/
│   ├── for_all_sensors_oled_masked_5min_5bands_jpeg.py
│   └── gps_cpp_style_c94_m8p.py
├── notebooks/
│   ├── flirplot_3sheet.ipynb
│   └── visualisation2.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Files

### `scripts/for_all_sensors_oled_masked_5min_5bands_jpeg.py`
Main sensor logging script. It combines:

- FLIR Lepton thermal frame capture
- radiometric TIFF saving
- 5 horizontal FLIR temperature bands
- JPEG thermal previews with band labels
- static-region masking after warmup
- HDC1080 ambient temperature and humidity
- u-blox GPS parsing
- SH1106 OLED display output
- LED health/status indicators
- CSV logging

### `scripts/gps_cpp_style_c94_m8p.py`
Standalone C94-M8P GPS logger in a C++-style parsing workflow. It records:

- validated GPS rows in `gps.csv`
- raw NMEA lines in `gps_raw.nmea`
- quality/debug rows in `gps_quality.csv`
- runtime logs in `system_log.txt`

### `notebooks/flirplot_3sheet.ipynb`
Notebook for creating Plotly dashboards from Excel sheets such as FLIR analysis sheets.

### `notebooks/visualisation2.ipynb`
Notebook for sensor/GPS visualisation and data analysis.

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some hardware-specific libraries are intended for Raspberry Pi / sensor hardware and may not install or run correctly on a normal laptop.

## Usage

Run the full environmental logger:

```bash
python scripts/for_all_sensors_oled_masked_5min_5bands_jpeg.py
```

Run the standalone GPS logger:

```bash
python scripts/gps_cpp_style_c94_m8p.py
```

Open notebooks:

```bash
jupyter notebook notebooks/flirplot_3sheet.ipynb
jupyter notebook notebooks/visualisation2.ipynb
```

## GitHub upload

```bash
git init
git add .
git commit -m "Initial commit: environmental monitoring scripts and notebooks"
git branch -M main
git remote add origin https://github.com/elmhmohammadi86/Environmental_Monitoring.git
git push -u origin main
```

If the repository already contains commits, use:

```bash
git pull origin main --rebase
git push -u origin main
```
