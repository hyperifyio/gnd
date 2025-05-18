#!/usr/bin/env bash
#
# normalize-as-ansi-text-file.sh - convert a UTF-8 file to basic ASCII via sed.
# Usage:  ./normalize-as-ansi-text-file.sh path/to/file.gnd
set -e
set -x

FILE="$1"

if test -f "$FILE"; then
  :
else
  echo "ERROR: No such file: $FILE" >&2
  exit 1
fi

if test -f "$FILE.bak"; then
  echo "ERROR: Working file exists already: $FILE.bak" >&2
  exit 2
fi

if iconv -f UTF-8 -t ISO-8859-1 "$FILE" 2> /dev/null > /dev/null; then
  :
else

  sed \
    -e 's/→/->/g'   \
    -e 's/←/<-/g'   \
    -e 's/“/"/g'    \
    -e 's/”/"/g'    \
    -e 's/‘/'\''/g' \
    -e 's/’/'\''/g' \
    -e 's/…/.../g'  \
    -e 's/—/--/g'   \
    -e 's/–/-/g'    \
    -e 's/•/*/g'    \
    -e 's/±/+\/-/g' \
    -e 's/×/x/g'    \
    -e 's/⁻/\^- /g' \
    -e 's/⁰/\^0/g'  \
    -e 's/¹/\^1/g'  \
    -e 's/²/\^2/g'  \
    -e 's/³/\^3/g'  \
    -e 's/⁴/\^4/g'  \
    -e 's/⁵/\^5/g'  \
    -e 's/⁶/\^6/g'  \
    -e 's/⁷/\^7/g'  \
    -e 's/⁸/\^8/g'  \
    -e 's/⁹/\^9/g'  \
    "$FILE" > "$FILE.bak"

  if iconv -f UTF-8 -t ISO-8859-1 "$FILE.bak" 2> /dev/null > /dev/null; then
    mv "$FILE.bak" "$FILE"
  else
    echo "ERROR: Could not normalize the file:" >&2
    iconv -f UTF-8 -t ISO-8859-1 "$FILE.bak" > /dev/null || true
    rm -f "$FILE.bak"
    exit 3
  fi

fi

