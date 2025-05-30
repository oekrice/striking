#!/bin/bash

cd /home/opc/striking || exit 1

# Stage, commit, and push
git add .
git commit -m "Daily auto-commit: $(date '+%Y-%m-%d %H:%M:%S')" || exit 0
git push origin data_backups
