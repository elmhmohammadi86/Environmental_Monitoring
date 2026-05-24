#!/usr/bin/env python3

import os
import csv
import time
import serial
from datetime import datetime


# ============================ CONFIG ============================
MACHINE_ID = "ROLLER1"
BOX_ID = "SMARTBOX1"

GPS_DEV = "/dev/ttyACM0"
GPS_BAUD = 9600
# ===============================================================


def mono_ns():
    return time.monotonic_ns()


def real_ns():
    return time.time_ns()


def iso_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def compact_time_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def nmea_to_decimal_degrees(value, hemi):
    if value == "":
        return ""

    try:
        raw = float(value)
    except ValueError:
        return ""

    deg = int(raw / 100.0)
    minutes = raw - deg * 100.0
    dec = deg + minutes / 60.0

    if hemi in ("S", "W"):
        dec = -dec

    # C++ با std::to_string حدوداً 6 رقم می‌دهد،
    # ولی اینجا 10 رقم نگه می‌داریم تا دقت از بین نرود.
    return f"{dec:.10f}"


def strip_checksum(value):
    if "*" in value:
        return value.split("*")[0]
    return value


def verify_nmea_checksum(line):
    if not line.startswith("$") or "*" not in line:
        return False

    try:
        body, checksum = line[1:].split("*", 1)
        checksum = checksum[:2]

        calc = 0
        for ch in body:
            calc ^= ord(ch)

        return calc == int(checksum, 16)
    except Exception:
        return False


def parse_nmea_line_like_cpp(line):
    result = {
        "sentence_type": "",
        "utc_time": "",
        "fix_status": "",
        "sats": "",
        "lat": "",
        "lon": "",
        "alt": "",
        "speed_knots": "",
        "course_deg": "",
        "raw": line,
        "hdop": "",
        "valid": False,
        "reason": "",
    }

    f = line.split(",")
    if not f:
        result["reason"] = "EMPTY"
        return result

    result["sentence_type"] = f[0]

    # ---------------- GGA ----------------
    # $GPGGA,time,lat,N,lon,E,fix,sats,hdop,alt,M,...
    # $GNGGA,time,lat,N,lon,E,fix,sats,hdop,alt,M,...
    if "GGA" in f[0] and len(f) >= 10:
        result["utc_time"] = f[1]
        result["fix_status"] = f[6]
        result["sats"] = f[7]
        result["hdop"] = f[8]
        result["alt"] = f[9]

        result["lat"] = nmea_to_decimal_degrees(f[2], f[3])
        result["lon"] = nmea_to_decimal_degrees(f[4], f[5])

        if result["fix_status"] == "0":
            result["reason"] = "GGA_INVALID_FIX"
            return result

        if result["lat"] == "" or result["lon"] == "":
            result["reason"] = "GGA_NO_POSITION"
            return result

        result["valid"] = True
        result["reason"] = "OK"
        return result

    # ---------------- RMC ----------------
    # $GPRMC,time,status,lat,N,lon,E,speed,course,date,...
    # $GNRMC,time,status,lat,N,lon,E,speed,course,date,...
    if "RMC" in f[0] and len(f) >= 9:
        result["utc_time"] = f[1]
        result["fix_status"] = f[2]
        result["speed_knots"] = f[7]
        result["course_deg"] = strip_checksum(f[8])

        result["lat"] = nmea_to_decimal_degrees(f[3], f[4])
        result["lon"] = nmea_to_decimal_degrees(f[5], f[6])

        if result["fix_status"] != "A":
            result["reason"] = "RMC_INVALID_STATUS"
            return result

        if result["lat"] == "" or result["lon"] == "":
            result["reason"] = "RMC_NO_POSITION"
            return result

        result["valid"] = True
        result["reason"] = "OK"
        return result

    result["reason"] = "IGNORED_SENTENCE"
    return result


def main():
    dataset_dir = f"dataset_{compact_time_now()}_{MACHINE_ID}_{BOX_ID}"
    os.makedirs(dataset_dir, exist_ok=True)

    gps_csv_path = os.path.join(dataset_dir, "gps.csv")
    raw_path = os.path.join(dataset_dir, "gps_raw.nmea")
    quality_path = os.path.join(dataset_dir, "gps_quality.csv")
    log_path = os.path.join(dataset_dir, "system_log.txt")

    gps_header = [
        "t_mono_ns",
        "t_real_ns",
        "machine_id",
        "box_id",
        "sentence_type",
        "utc_time",
        "fix_status",
        "sats",
        "lat",
        "lon",
        "alt",
        "speed_knots",
        "course_deg",
        "raw",
    ]

    quality_header = [
        "t_mono_ns",
        "t_real_ns",
        "sentence_type",
        "valid",
        "reason",
        "fix_status",
        "sats",
        "hdop",
        "lat",
        "lon",
        "alt",
        "speed_knots",
        "course_deg",
        "raw",
    ]

    print("C94-M8P GPS logger, C++ style")
    print(f"Port: {GPS_DEV}")
    print(f"Baud: {GPS_BAUD}")
    print(f"Output: {dataset_dir}")
    print("Stop with Ctrl+C")
    print()

    with open(gps_csv_path, "w", newline="") as gps_file, \
         open(raw_path, "w") as raw_file, \
         open(quality_path, "w", newline="") as quality_file, \
         open(log_path, "w") as log_file:

        gps_writer = csv.DictWriter(gps_file, fieldnames=gps_header)
        quality_writer = csv.DictWriter(quality_file, fieldnames=quality_header)

        gps_writer.writeheader()
        quality_writer.writeheader()

        log_file.write(f"{iso_time_now()} [INFO] LOGGER_START dataset={dataset_dir}\n")
        log_file.flush()

        try:
            ser = serial.Serial(
                GPS_DEV,
                GPS_BAUD,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
        except Exception as e:
            log_file.write(f"{iso_time_now()} [ERROR] GPS_OPEN_FAILED {GPS_DEV}: {e}\n")
            log_file.flush()
            print(f"ERROR: cannot open {GPS_DEV}: {e}")
            return

        gps_rows = 0
        raw_rows = 0
        invalid_rows = 0
        bad_checksum_rows = 0
        last_print = time.time()

        try:
            while True:
                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("ascii", errors="ignore").strip()

                if not line.startswith("$"):
                    continue

                raw_file.write(line + "\n")
                raw_file.flush()
                raw_rows += 1

                # فقط مثل C++: GGA و RMC
                if "GGA" not in line and "RMC" not in line:
                    continue

                t_mono = mono_ns()
                t_real = real_ns()

                if not verify_nmea_checksum(line):
                    bad_checksum_rows += 1
                    log_file.write(f"{iso_time_now()} [WARN] BAD_CHECKSUM {line}\n")
                    log_file.flush()
                    continue

                parsed = parse_nmea_line_like_cpp(line)

                quality_writer.writerow({
                    "t_mono_ns": t_mono,
                    "t_real_ns": t_real,
                    "sentence_type": parsed["sentence_type"],
                    "valid": int(parsed["valid"]),
                    "reason": parsed["reason"],
                    "fix_status": parsed["fix_status"],
                    "sats": parsed["sats"],
                    "hdop": parsed["hdop"],
                    "lat": parsed["lat"],
                    "lon": parsed["lon"],
                    "alt": parsed["alt"],
                    "speed_knots": parsed["speed_knots"],
                    "course_deg": parsed["course_deg"],
                    "raw": parsed["raw"],
                })
                quality_file.flush()

                if not parsed["valid"]:
                    invalid_rows += 1
                    continue

                gps_writer.writerow({
                    "t_mono_ns": t_mono,
                    "t_real_ns": t_real,
                    "machine_id": MACHINE_ID,
                    "box_id": BOX_ID,
                    "sentence_type": parsed["sentence_type"],
                    "utc_time": parsed["utc_time"],
                    "fix_status": parsed["fix_status"],
                    "sats": parsed["sats"],
                    "lat": parsed["lat"],
                    "lon": parsed["lon"],
                    "alt": parsed["alt"],
                    "speed_knots": parsed["speed_knots"],
                    "course_deg": parsed["course_deg"],
                    "raw": parsed["raw"],
                })
                gps_file.flush()
                gps_rows += 1

                now = time.time()
                if now - last_print >= 1.0:
                    last_print = now

                    print(
                        f"[{iso_time_now()}] "
                        f"GPS={gps_rows} raw={raw_rows} "
                        f"invalid={invalid_rows} bad_checksum={bad_checksum_rows} | "
                        f"type={parsed['sentence_type']} "
                        f"fix={parsed['fix_status']} "
                        f"sats={parsed['sats']} "
                        f"hdop={parsed['hdop']} "
                        f"lat={parsed['lat']} lon={parsed['lon']}"
                    )

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            ser.close()
            log_file.write(f"{iso_time_now()} [INFO] LOGGER_STOPPED\n")
            log_file.write(f"{iso_time_now()} [INFO] GPS_ROWS={gps_rows}\n")
            log_file.write(f"{iso_time_now()} [INFO] RAW_ROWS={raw_rows}\n")
            log_file.write(f"{iso_time_now()} [INFO] INVALID_ROWS={invalid_rows}\n")
            log_file.write(f"{iso_time_now()} [INFO] BAD_CHECKSUM_ROWS={bad_checksum_rows}\n")
            log_file.flush()

    print(f"Dataset saved in: {dataset_dir}")


if __name__ == "__main__":
    main()