#!/bin/bash

# === CONFIG ===
SOURCE_DIR="/home/opc/striking/frequency_data"
REMOTE_DIR="brenda_backup:frequency_data"
RCLONE_CONFIG="/home/opc/.config/rclone/rclone.conf"
LOG="/home/opc/rclone_backup.log"

# === BEGIN LOGGING ===
echo "[$(date)] Backup started" >> "$LOG"

# === CHECK IF SOURCE EXISTS ===
if [ ! -d "$SOURCE_DIR" ]; then
    echo "[$(date)] ERROR: Source directory not found: $SOURCE_DIR" >> "$LOG"
    exit 1
fi

# === SKIP IF EMPTY ===
if [ -z "$(ls -A "$SOURCE_DIR")" ]; then
    echo "[$(date)] WARNING: Source directory is empty — skipping backup" >> "$LOG"
    exit 0
fi

# === COPY TO GOOGLE DRIVE (overwrite) ===
rclone --config="$RCLONE_CONFIG" copy "$SOURCE_DIR" "$REMOTE_DIR" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Upload failed" >> "$LOG"
    exit 1
fi

echo "[$(date)] Backup completed successfully" >> "$LOG"
