#!/bin/bash

# === CONFIG ===
SOURCE_DIR="frequency_data"       # Replace with your actual data path
REMOTE_NAME="brenda_backup"                # Your rclone remote name
DEST_PATH="frequency_data"              # Folder in Google Drive (rclone will create if not exists)
DATE=$(date +%F_%H-%M)
ARCHIVE="/tmp/frequency_backup_$DATE.tar.gz"

# === BACKUP ===
tar -czf "$ARCHIVE" -C "$SOURCE_DIR" .
rclone copy "$ARCHIVE" "$REMOTE_NAME:$DEST_PATH"
rm -f "$ARCHIVE"
