#!/usr/bin/env python3
import os
import datetime
from pathlib import Path

def parse_time_from_line(line: str) -> datetime.datetime | None:
    """Extract datetime from a line like 'Start time: 2026-03-31 20:10:18'."""
    line = line.strip()
    if not line:
        return None
    # Split at the first colon
    parts = line.split(':', 1)
    if len(parts) != 2:
        return None
    time_str = parts[1].strip()
    # Expected format: YYYY-MM-DD HH:MM:SS
    try:
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("failed1")

def process_file(file_path: Path) -> tuple[str, datetime.timedelta] | None:
    """Return (filename, time_difference) if successful, otherwise None."""
    start_time = None
    end_time = None


    with open(file_path, 'r') as f:
        for line in f:
            lower_line = line.lower()
            if lower_line.startswith('start:'):
                start_time = parse_time_from_line(line)
            elif lower_line.startswith('finished at:'):
                end_time = parse_time_from_line(line)
            # Early exit if both are found
            if start_time and end_time:
                break

    if start_time and end_time:
        return file_path.name, end_time - start_time
    return None

def main():
    folder = Path("../SEEGNet2 results/temp")
    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder '{folder}' does not exist or is not a directory.")

    for item in folder.iterdir():
        if item.is_file():
            result = process_file(item)
            if result:
                filename, diff = result
                print(f"{filename}: {diff}")

if __name__ == "__main__":
    main()