#!/usr/bin/env bash
#
# colorbin.sh: run `xxd -b` on a file, then color each 2-bit group
#   00 →blued, 01 → green, 10 → yellow, 11red
#
if [[ -z "$1" ]]; then
  echo "Usage: $0 <binary-file> START_OFFSET SIZE"
  exit 1
fi

START_OFFSET=$2
LENGTH=$3

xxd -b -s "$START_OFFSET" -l "$LENGTH" "$1" \
| awk '
  # ANSI‐escape definitions: change these if you want other colors
  BEGIN {
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    RESET  = "\033[0m"
  }
  {
    # For each field in the xxd output:
    #  — binary‐bytes appear in fields that are exactly 8 chars of 0/1
    #  — everything else (offset, hex, ASCII) we print unchanged.
    out_line = ""
    for (i = 1; i <= NF; i++) {
      field = $i
      if (field ~ /^[01]{8}$/) {
        # This is an 8-bit binary chunk.  Split into four 2-bit subfields:
        colored = ""
        for (j = 1; j <= 8; j += 2) {
          bits = substr(field, j, 2)
          if      (bits == "00") col = BLUE
          else if (bits == "01") col = GREEN
          else if (bits == "10") col = YELLOW
          else                   col = RED
          colored = colored col bits RESET
        }
        out_line = out_line colored " "
      } else {
        # Not raw binary—just print it literally (e.g. offset/hex/ASCII):
        out_line = out_line field " "
      }
    }
    # Trim trailing space and print
    sub(/[[:space:]]$/, "", out_line)
    print out_line
  }
'

