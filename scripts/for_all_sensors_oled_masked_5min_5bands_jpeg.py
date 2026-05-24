import time
import csv
import board
import busio
import numpy as np
import threading
import subprocess
from flirpy.camera.lepton import Lepton
from PIL import Image, ImageDraw, ImageFont
from adafruit_bus_device.i2c_device import I2CDevice
from datetime import datetime
import serial
import pynmea2
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import tifffile

# OLED (SH1106)
from luma.core.interface.serial import i2c as luma_i2c
from luma.oled.device import sh1106

# LED
from digitalio import DigitalInOut, Direction

# ===========================
# SETTINGS
# ===========================
SAVE_EVERY_SEC = 10

THERMAL_ROOT_DIR = "./thermal_frames"
os.makedirs(THERMAL_ROOT_DIR, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
THERMAL_RUN_DIR = os.path.join(THERMAL_ROOT_DIR, f"run_{RUN_TIMESTAMP}")
os.makedirs(THERMAL_RUN_DIR, exist_ok=True)

CMAP_NAME = "inferno"
AUTO_COLOR_RANGE = True
USE_PERCENTILE_RANGE = True
P_LOW = 2.0
P_HIGH = 98.0

# Static-region calibration (build once after warmup)
MASK_WARMUP_SEC = 300
STATIC_STD_PERCENTILE = 30.0
STATIC_RANGE_PERCENTILE = 30.0
STATIC_MIN_PIXELS = 20
MIN_VALID_WARMUP_FRAMES = 30
MASK_PREVIEW_ALPHA = 0.45

# 5-band FLIR analysis and JPEG saving
# Banding is horizontal: the FLIR image is split from top to bottom into 5 parallel bands.
BAND_COUNT = 5
JPEG_SAVE_EVERY_SEC = SAVE_EVERY_SEC
JPEG_QUALITY = 95
BAND_LINE_WIDTH = 2

# ===========================
# PURETHERMAL MINI: FFC HELPERS
# ===========================
VIDEO_DEV_CANDIDATES = ['/dev/video0', '/dev/video1']
PT1_XML = os.path.expanduser('~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml')
FFC_INTERVAL_SEC = 180
DISCARD_AFTER_FFC_FRAMES = 3

_ffc_device = None
_last_ffc_time = 0.0


def _v4l2_list_controls(dev: str) -> str:
    try:
        r = subprocess.run(['v4l2-ctl', '-d', dev, '-l'], capture_output=True, text=True)
        return r.stdout or ''
    except Exception:
        return ''


def _has_run_ffc(dev: str) -> bool:
    return 'lep_cid_rad_run_ffc' in _v4l2_list_controls(dev)


def _try_import_pt1_xml(dev: str) -> bool:
    if not os.path.exists(PT1_XML):
        return False
    try:
        subprocess.run(
            ['uvcdynctrl', '-d', dev, '-i', PT1_XML],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def _detect_ffc_device() -> str | None:
    for dev in VIDEO_DEV_CANDIDATES:
        if _has_run_ffc(dev):
            return dev
    for dev in VIDEO_DEV_CANDIDATES:
        _try_import_pt1_xml(dev)
    for dev in VIDEO_DEV_CANDIDATES:
        if _has_run_ffc(dev):
            return dev
    return None


def run_ffc_if_available(force: bool = False) -> bool:
    global _ffc_device, _last_ffc_time
    now = time.time()
    if (not force) and _last_ffc_time and (now - _last_ffc_time) < FFC_INTERVAL_SEC:
        return False

    if _ffc_device is None:
        _ffc_device = _detect_ffc_device()

    if _ffc_device is None:
        logging.warning('FFC control not available (lep_cid_rad_run_ffc not mapped).')
        return False

    try:
        r = subprocess.run(
            ['v4l2-ctl', '-d', _ffc_device, '-c', 'lep_cid_rad_run_ffc=1'],
            capture_output=True,
            text=True
        )
        if r.returncode != 0:
            logging.warning(f'FFC trigger failed on {_ffc_device}: {r.stderr.strip()}')
            return False
        _last_ffc_time = now
        logging.info(f'FFC triggered on {_ffc_device}')
        return True
    except Exception as e:
        logging.warning(f'FFC trigger exception: {e}')
        return False


def thermal_to_rgb(temp_c, vmin, vmax):
    if vmax <= vmin:
        vmax = vmin + 1e-6
    cmap = cm.get_cmap(CMAP_NAME)
    norm = np.clip((temp_c - vmin) / (vmax - vmin), 0.0, 1.0)
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


def choose_color_range(temp_c, tmin, tmax):
    if not AUTO_COLOR_RANGE:
        return 20.0, 40.0
    if USE_PERCENTILE_RANGE:
        return float(np.percentile(temp_c, P_LOW)), float(np.percentile(temp_c, P_HIGH))
    return tmin, tmax


def init_mask_stats(frame_shape):
    return {
        'count': 0,
        'sum': np.zeros(frame_shape, dtype=np.float64),
        'sum_sq': np.zeros(frame_shape, dtype=np.float64),
        'min': np.full(frame_shape, np.inf, dtype=np.float64),
        'max': np.full(frame_shape, -np.inf, dtype=np.float64),
        'reference_temp_c': None,
        'reference_timestamp': None,
    }


def update_mask_stats(stats, temp_c, timestamp_str):
    temp = temp_c.astype(np.float64)
    stats['count'] += 1
    stats['sum'] += temp
    stats['sum_sq'] += temp * temp
    stats['min'] = np.minimum(stats['min'], temp)
    stats['max'] = np.maximum(stats['max'], temp)
    stats['reference_temp_c'] = temp.copy()
    stats['reference_timestamp'] = timestamp_str


def smooth_mask(mask, passes=1):
    result = mask.astype(bool)
    for _ in range(passes):
        padded = np.pad(result.astype(np.uint8), 1, mode='edge')
        neighbors = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        result = neighbors >= 5
    return result


def build_static_mask_from_stats(stats):
    count = stats['count']
    if count < MIN_VALID_WARMUP_FRAMES:
        return None

    mean_map = stats['sum'] / count
    variance = np.maximum((stats['sum_sq'] / count) - (mean_map * mean_map), 0.0)
    std_map = np.sqrt(variance)
    range_map = stats['max'] - stats['min']

    std_thr = float(np.percentile(std_map, STATIC_STD_PERCENTILE))
    range_thr = float(np.percentile(range_map, STATIC_RANGE_PERCENTILE))

    static_mask = (std_map <= std_thr) & (range_map <= range_thr)
    static_mask = smooth_mask(static_mask, passes=1)

    if int(np.count_nonzero(static_mask)) < STATIC_MIN_PIXELS:
        return None

    return static_mask


def compute_masked_average(temp_c, static_mask):
    if static_mask is None:
        return "NA"

    valid_mask = ~static_mask
    valid_pixels = temp_c[valid_mask]

    if valid_pixels.size == 0:
        return "NA"

    return round(float(np.mean(valid_pixels)), 2)


def compute_flir_band_averages(temp_c, band_count=BAND_COUNT):
    """
    Split the FLIR temperature frame into horizontal bands and return:
      - band_averages: list of average temperatures for each band
      - band_ranges: list of (y_start, y_end) pixel ranges for each band

    The last band automatically includes any leftover rows if image height
    is not perfectly divisible by band_count.
    """
    if temp_c is None:
        return ["NA"] * band_count, []

    h, w = temp_c.shape
    y_edges = np.linspace(0, h, band_count + 1, dtype=int)

    band_averages = []
    band_ranges = []

    for i in range(band_count):
        y0 = int(y_edges[i])
        y1 = int(y_edges[i + 1])
        band_ranges.append((y0, y1))

        band_pixels = temp_c[y0:y1, :]
        if band_pixels.size == 0:
            band_averages.append("NA")
        else:
            band_averages.append(round(float(np.mean(band_pixels)), 2))

    return band_averages, band_ranges


def save_flir_jpeg_with_bands(temp_c, band_averages, band_ranges, timestamp_str, output_path):
    """Save a color JPEG preview of the FLIR frame with 5-band divisions drawn on top."""
    if temp_c is None:
        return False

    tmin = float(np.min(temp_c))
    tmax = float(np.max(temp_c))
    vmin, vmax = choose_color_range(temp_c, tmin, tmax)
    thermal_rgb = thermal_to_rgb(temp_c, vmin, vmax)

    img = Image.fromarray(thermal_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    w, h = img.size

    # Draw horizontal band separator lines and labels.
    for i, (y0, y1) in enumerate(band_ranges):
        if i > 0:
            draw.line([(0, y0), (w - 1, y0)], fill=(255, 255, 255), width=BAND_LINE_WIDTH)

        label_temp = band_averages[i] if i < len(band_averages) else "NA"
        label = f"B{i + 1}: {label_temp} C"
        text_x = 2
        text_y = max(y0 + 2, 0)

        # Small black background behind text for readability.
        try:
            bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
        except Exception:
            draw.rectangle((text_x, text_y, text_x + 70, text_y + 10), fill=(0, 0, 0))

        draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))

    # Draw outer border and timestamp.
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(255, 255, 255), width=1)
    ts_label = timestamp_str
    try:
        bbox = draw.textbbox((2, h - 12), ts_label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
    except Exception:
        draw.rectangle((2, h - 12, w - 1, h - 1), fill=(0, 0, 0))
    draw.text((2, h - 12), ts_label, font=font, fill=(255, 255, 255))

    img.save(output_path, format="JPEG", quality=JPEG_QUALITY)
    return True


def save_static_mask_visuals(temp_c, static_mask, timestamp_str, created_elapsed_s):
    if temp_c is None or static_mask is None:
        return

    tmin = float(np.min(temp_c))
    tmax = float(np.max(temp_c))
    vmin, vmax = choose_color_range(temp_c, tmin, tmax)
    thermal_rgb = thermal_to_rgb(temp_c, vmin, vmax)

    ref_img = Image.fromarray(thermal_rgb)
    mask_img = Image.fromarray((static_mask.astype(np.uint8) * 255), mode='L')

    overlay_rgb = thermal_rgb.copy()
    overlay_rgb[static_mask] = (
        overlay_rgb[static_mask].astype(np.float32) * (1.0 - MASK_PREVIEW_ALPHA)
        + np.array([0, 255, 255], dtype=np.float32) * MASK_PREVIEW_ALPHA
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay_rgb)

    label = f"mask_created_at_{created_elapsed_s}s"
    ref_path = os.path.join(THERMAL_RUN_DIR, f"{label}_reference.png")
    mask_path = os.path.join(THERMAL_RUN_DIR, f"{label}_mask.png")
    overlay_path = os.path.join(THERMAL_RUN_DIR, f"{label}_overlay.png")
    info_path = os.path.join(THERMAL_RUN_DIR, f"{label}_info.txt")

    ref_img.save(ref_path)
    mask_img.save(mask_path)
    overlay_img.save(overlay_path)

    with open(info_path, 'w') as f:
        f.write(f"Mask created timestamp: {timestamp_str}\n")
        f.write(f"Mask created elapsed seconds: {created_elapsed_s}\n")
        f.write(f"Masked pixels: {int(np.count_nonzero(static_mask))}\n")
        f.write(f"Unmasked pixels: {int(np.count_nonzero(~static_mask))}\n")
        f.write(f"Reference frame shape: {temp_c.shape}\n")
        f.write(f"Reference frame dtype: {temp_c.dtype}\n")
        f.write(f"Color range used: vmin={vmin:.2f}, vmax={vmax:.2f}\n")

    logging.info(f"Saved mask reference image: {ref_path}")
    logging.info(f"Saved mask image: {mask_path}")
    logging.info(f"Saved mask overlay image: {overlay_path}")
    logging.info(f"Saved mask info: {info_path}")


# ===========================
# LOGGING
# ===========================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "logger.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("Program started")
logging.info(f"Thermal raw TIFF/JPEG directory: {THERMAL_RUN_DIR}")

# ===========================
# LED SETUP
# ===========================
LED_FLIR = board.D21
LED_HDC  = board.D20
LED_GPS  = board.D16

led_flir = DigitalInOut(LED_FLIR)
led_hdc  = DigitalInOut(LED_HDC)
led_gps  = DigitalInOut(LED_GPS)

for led in [led_flir, led_hdc, led_gps]:
    led.direction = Direction.OUTPUT
    led.value = False


def blink_led(led, event):
    state = False
    while True:
        if event.is_set():
            state = not state
            led.value = state
            time.sleep(0.3)
        else:
            if led.value:
                led.value = False
            time.sleep(0.1)


flir_event = threading.Event()
hdc_event  = threading.Event()
gps_event  = threading.Event()

threading.Thread(target=blink_led, args=(led_flir, flir_event), daemon=True).start()
threading.Thread(target=blink_led, args=(led_hdc, hdc_event), daemon=True).start()
threading.Thread(target=blink_led, args=(led_gps, gps_event), daemon=True).start()

# ===========================
# GPS CLASS
# ===========================
class Ublox:
    _NODATA = -100

    def __init__(self):
        self.serial_port = None
        self.latitude = self.longitude = self.altitude = self.speed = self.satellites = self.hdop = self._NODATA
        self.timestamp = "NA"
        self.lock = threading.Lock()
        self.last_update_monotonic = None

    def initialize(self, com_port='/dev/ttyACM0', baud_rate=9600):
        try:
            self.serial_port = serial.Serial(com_port, baud_rate, timeout=1)
            return self.serial_port.is_open
        except Exception as e:
            logging.warning(f"GPS init error: {e}")
            return False

    def update_loop(self):
        while True:
            try:
                line = self.serial_port.readline().decode('ascii', errors='replace').strip()
                if not line.startswith('$'):
                    continue

                if line.startswith(('$GPGGA', '$GNGGA', '$GPRMC', '$GNRMC')):
                    try:
                        msg = pynmea2.parse(line)
                    except pynmea2.ParseError:
                        continue

                    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

                    with self.lock:
                        self.timestamp = now
                        self.latitude = getattr(msg, 'latitude', self._NODATA)
                        self.longitude = getattr(msg, 'longitude', self._NODATA)
                        self.altitude = getattr(msg, 'altitude', self._NODATA)
                        self.satellites = getattr(msg, 'num_sats', self._NODATA)
                        self.hdop = getattr(msg, 'horizontal_dil', self._NODATA)

                        if hasattr(msg, 'spd_over_grnd') and msg.spd_over_grnd:
                            try:
                                self.speed = round(float(msg.spd_over_grnd) * 1.852, 2)
                            except ValueError:
                                self.speed = self._NODATA

                        self.last_update_monotonic = time.monotonic()
            except Exception as e:
                logging.warning(f"GPS read error: {e}")
                time.sleep(0.2)

    def start_thread(self):
        t = threading.Thread(target=self.update_loop, daemon=True)
        t.start()


# ===========================
# SENSOR INIT
# ===========================
i2c = busio.I2C(board.SCL, board.SDA)

# OLED INIT - SH1106 @ 0x3C
try:
    OLED_ADDR = 0x3C
    oled_serial = luma_i2c(port=1, address=OLED_ADDR)
    oled = sh1106(oled_serial)
    oled_enabled = True

    boot_img = Image.new("1", (oled.width, oled.height))
    boot_draw = ImageDraw.Draw(boot_img)
    boot_font = ImageFont.load_default()
    boot_draw.text((0, 0), "OLED SH1106 OK", font=boot_font, fill=255)
    oled.display(boot_img)

    logging.info("OLED initialized successfully at 0x3C")
except Exception as e:
    logging.warning(f"OLED init failed: {e}")
    oled_enabled = False

# HDC1080 INIT
HDC1080_ADDR = 0x40
try:
    hdc_device = I2CDevice(i2c, HDC1080_ADDR)
    hdc_enabled = True
    logging.info("HDC1080 initialized successfully")
except Exception as e:
    logging.warning(f"HDC1080 init failed: {e}")
    hdc_enabled = False

# FLIR INIT
cam = None
flir_enabled = False
REINIT_COOLDOWN = 3.0
_last_flir_try = 0.0
static_region_mask = None
static_mask_ready = False
mask_created_elapsed_s = "NA"
mask_stats = None


def init_flir():
    global cam, flir_enabled
    try:
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
            cam = None

        cam = Lepton()
        flir_enabled = True
        logging.info("FLIR ready")
    except Exception as e:
        cam = None
        flir_enabled = False
        logging.warning(f"FLIR init failed: {e}")


init_flir()

try:
    if run_ffc_if_available(force=True):
        time.sleep(0.4)
except Exception:
    pass

# GPS INIT
gps = Ublox()
if gps.initialize('/dev/ttyACM0', 9600):
    gps.start_thread()
    logging.info("GPS thread started")
    gps_enabled = True
else:
    logging.warning("GPS initialization failed")
    gps_enabled = False

# ===========================
# CSV LOGGING
# ===========================
data_dir = "./data_logs"
os.makedirs(data_dir, exist_ok=True)

csv_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(data_dir, f"combined_data_{csv_timestamp}.csv")
file = open(filename, 'w', newline='')
writer = csv.writer(file)

writer.writerow([
    "Time (s)",
    "Timestamp",
    "FLIR Center 2x2 Temp (°C)",
    "FLIR Full Avg Temp (°C)",
    "FLIR Masked Avg Temp (°C)",
    "FLIR Band 1 Avg Temp (°C)",
    "FLIR Band 2 Avg Temp (°C)",
    "FLIR Band 3 Avg Temp (°C)",
    "FLIR Band 4 Avg Temp (°C)",
    "FLIR Band 5 Avg Temp (°C)",
    "Mask Ready",
    "Mask Created At (s)",
    "Ambient Temp (°C)",
    "Humidity (%)",
    "Latitude",
    "Longitude",
    "Altitude (m)",
    "Speed (km/h)",
    "Satellites",
    "HDOP"
])

start_time = time.time()
logging.info(f"Data logging started: {filename}")
logging.info(f"Masked average warmup duration: {MASK_WARMUP_SEC} seconds")

GPS_STALE_SEC = 5
last_save = 0

try:
    while True:
        elapsed = int(time.time() - start_time)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        flir_temp = "NA"
        full_avg_temp = "NA"
        masked_avg_temp = "NA"
        band_avg_temps = ["NA"] * BAND_COUNT
        band_ranges = []
        temp_env = humidity = lat = lon = alt = speed = sats = hdop = "NA"

        raw_frame = None
        temp_c = None

        now_mono = time.monotonic()

        if not flir_enabled and (now_mono - _last_flir_try) > REINIT_COOLDOWN:
            _last_flir_try = now_mono
            init_flir()

        # ===== FLIR =====
        flir_ok = flir_enabled

        if flir_enabled:
            try:
                did_ffc = run_ffc_if_available(force=False)
                if did_ffc:
                    for _ in range(DISCARD_AFTER_FFC_FRAMES):
                        try:
                            cam.grab()
                        except Exception:
                            pass
                        time.sleep(0.02)

                image = cam.grab()
                raw_frame = image.astype(np.uint16)
                raw = image.astype(np.int32)
                temp_c = raw / 100.0 - 273.15

                if mask_stats is None:
                    mask_stats = init_mask_stats(temp_c.shape)

                h, w = temp_c.shape
                cy, cx = h // 2, w // 2

                flir_temp = round(float(np.mean(temp_c[cy-1:cy+1, cx-1:cx+1])), 2)
                full_avg_temp = round(float(np.mean(temp_c)), 2)
                band_avg_temps, band_ranges = compute_flir_band_averages(temp_c, BAND_COUNT)

                if not static_mask_ready and elapsed < MASK_WARMUP_SEC:
                    update_mask_stats(mask_stats, temp_c, timestamp)
                elif not static_mask_ready:
                    static_region_mask = build_static_mask_from_stats(mask_stats)
                    static_mask_ready = static_region_mask is not None
                    if static_mask_ready:
                        mask_created_elapsed_s = elapsed
                        save_static_mask_visuals(
                            mask_stats['reference_temp_c'],
                            static_region_mask,
                            mask_stats['reference_timestamp'] or timestamp,
                            mask_created_elapsed_s
                        )
                        logging.info(
                            f"Static FLIR mask created after warmup. "
                            f"Warmup frames: {mask_stats['count']}, "
                            f"Masked pixels: {int(np.count_nonzero(static_region_mask))}"
                        )
                    else:
                        logging.warning(
                            f"Static FLIR mask could not be created after {elapsed}s. "
                            f"Warmup frames collected: {mask_stats['count']}"
                        )

                if static_mask_ready:
                    masked_avg_temp = compute_masked_average(temp_c, static_region_mask)

                if (flir_temp > 150.0) or (flir_temp < -40.0):
                    raise ValueError(f'Glitch frame (center temp out of range): {flir_temp}')

                if (full_avg_temp > 150.0) or (full_avg_temp < -40.0):
                    raise ValueError(f'Glitch frame (full avg temp out of range): {full_avg_temp}')

                if masked_avg_temp != "NA":
                    if (masked_avg_temp > 150.0) or (masked_avg_temp < -40.0):
                        raise ValueError(f'Glitch frame (masked avg temp out of range): {masked_avg_temp}')

                for idx, band_temp in enumerate(band_avg_temps, start=1):
                    if band_temp != "NA" and ((band_temp > 150.0) or (band_temp < -40.0)):
                        raise ValueError(f'Glitch frame (band {idx} avg temp out of range): {band_temp}')

                now = time.time()
                if raw_frame is not None and now - last_save >= SAVE_EVERY_SEC:
                    tiff_name = os.path.join(
                        THERMAL_RUN_DIR,
                        f"flir_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff"
                    )

                    tifffile.imwrite(tiff_name, raw_frame)

                    jpg_name = tiff_name.replace("flir_raw_", "flir_bands_").replace(".tiff", ".jpg")
                    save_flir_jpeg_with_bands(temp_c, band_avg_temps, band_ranges, timestamp, jpg_name)

                    meta_name = tiff_name.replace(".tiff", ".txt")
                    with open(meta_name, "w") as f:
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Type: FLIR raw/radiometric frame\n")
                        f.write(f"Shape: {raw_frame.shape}\n")
                        f.write(f"Dtype: {raw_frame.dtype}\n")
                        f.write(f"Center_2x2_C: {flir_temp}\n")
                        f.write(f"Full_Avg_C: {full_avg_temp}\n")
                        f.write(f"Masked_Avg_C: {masked_avg_temp}\n")
                        for band_idx, band_temp in enumerate(band_avg_temps, start=1):
                            f.write(f"Band_{band_idx}_Avg_C: {band_temp}\n")
                        f.write(f"Band_Count: {BAND_COUNT}\n")
                        f.write(f"Band_Orientation: horizontal_top_to_bottom\n")
                        f.write(f"Band_JPEG: {jpg_name}\n")
                        f.write(f"Mask_Ready: {static_mask_ready}\n")
                        f.write(f"Mask_Created_At_s: {mask_created_elapsed_s}\n")
                        if mask_stats is not None:
                            f.write(f"Warmup_Frame_Count: {mask_stats['count']}\n")
                        if static_mask_ready:
                            f.write(f"Static_Masked_Pixels: {int(np.count_nonzero(static_region_mask))}\n")

                    logging.info(f"Saved raw FLIR TIFF: {tiff_name}")
                    logging.info(f"Saved FLIR band JPEG: {jpg_name}")
                    logging.info(f"Saved TIFF/JPEG metadata: {meta_name}")

                    last_save = now

            except Exception as e:
                logging.warning(f"FLIR error: {e}")
                flir_temp = "NA"
                full_avg_temp = "NA"
                masked_avg_temp = "NA"
                band_avg_temps = ["NA"] * BAND_COUNT
                band_ranges = []
                flir_ok = False
                flir_enabled = False

                try:
                    if cam is not None:
                        cam.close()
                except Exception:
                    pass
                cam = None

        # ===== HDC1080 =====
        hdc_ok = hdc_enabled
        if hdc_enabled:
            try:
                with hdc_device:
                    hdc_device.write(bytes([0x00]))
                    time.sleep(0.015)
                    temp_data = bytearray(2)
                    hdc_device.readinto(temp_data)
                    temp_env = round(((temp_data[0] << 8) | temp_data[1]) * (165.0 / 65536.0) - 40.0, 2)

                    hdc_device.write(bytes([0x01]))
                    time.sleep(0.015)
                    hum_data = bytearray(2)
                    hdc_device.readinto(hum_data)
                    humidity = round(((hum_data[0] << 8) | hum_data[1]) * (100.0 / 65536.0), 2)
            except Exception as e:
                logging.warning(f"HDC1080 error: {e}")
                temp_env = humidity = "NA"
                hdc_ok = False

        # ===== GPS =====
        gps_ok = gps_enabled
        if gps_enabled:
            last = gps.last_update_monotonic
            if (last is None) or ((time.monotonic() - last) > GPS_STALE_SEC):
                gps_ok = False
            with gps.lock:
                lat = round(gps.latitude, 6) if gps.latitude != -100 else "NA"
                lon = round(gps.longitude, 6) if gps.longitude != -100 else "NA"
                alt = gps.altitude if gps.altitude != -100 else "NA"
                speed = gps.speed if gps.speed != -100 else "NA"
                sats = gps.satellites if gps.satellites != -100 else "NA"
                hdop = gps.hdop if gps.hdop != -100 else "NA"

        # ===== LED CONTROL =====
        if flir_ok:
            flir_event.clear()
        else:
            flir_event.set()

        if hdc_ok:
            hdc_event.clear()
        else:
            hdc_event.set()

        if gps_ok:
            gps_event.clear()
        else:
            gps_event.set()

        # ===== CSV =====
        writer.writerow([
            elapsed,
            timestamp,
            flir_temp,
            full_avg_temp,
            masked_avg_temp,
            band_avg_temps[0],
            band_avg_temps[1],
            band_avg_temps[2],
            band_avg_temps[3],
            band_avg_temps[4],
            int(static_mask_ready),
            mask_created_elapsed_s,
            temp_env,
            humidity,
            lat,
            lon,
            alt,
            speed,
            sats,
            hdop
        ])
        file.flush()

        # ===== OLED =====
        if oled_enabled:
            try:
                image_oled = Image.new("1", (oled.width, oled.height))
                draw = ImageDraw.Draw(image_oled)
                font = ImageFont.load_default()

                lat_str = str(lat)
                lon_str = str(lon)
                flir_display_value = masked_avg_temp if masked_avg_temp != "NA" else full_avg_temp

                draw.text((0, 0),  f"FLIR:{flir_display_value}", font=font, fill=255)
                draw.text((0, 12), f"Air:{temp_env}C", font=font, fill=255)
                draw.text((0, 24), f"Hum:{humidity}%", font=font, fill=255)
                draw.text((0, 36), f"Lat:{lat_str[:14]}", font=font, fill=255)
                draw.text((0, 48), f"Lon:{lon_str[:14]}", font=font, fill=255)

                oled.display(image_oled)
            except Exception as e:
                logging.warning(f"OLED error: {e}")

        print(
            f"{elapsed}s -> "
            f"FLIR Center: {flir_temp} C | "
            f"FLIR Avg: {full_avg_temp} C | "
            f"FLIR Masked Avg: {masked_avg_temp} C | "
            f"Bands: {band_avg_temps} | "
            f"Mask Ready: {int(static_mask_ready)} | "
            f"Env: {temp_env} C | "
            f"Humidity: {humidity}% | "
            f"GPS: ({lat}, {lon}) | "
            f"Speed: {speed} km/h"
        )

        time.sleep(1)

except KeyboardInterrupt:
    logging.info("Logging stopped manually")

finally:
    for led in [led_flir, led_hdc, led_gps]:
        led.value = False

    try:
        file.close()
    except Exception:
        pass

    try:
        if cam is not None:
            cam.close()
    except Exception:
        pass

    logging.info("Resources cleaned up. Exiting...")
