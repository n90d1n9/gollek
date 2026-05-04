#!/bin/bash
echo "Finding Java files..."
files=$(find . -name "*.java" -type f)
total=$(echo "$files" | wc -l)
echo "Found $total Java files. Scanning for usage..."

unused=()
for f in $files; do
  classname=$(basename "$f" .java)
  
  # Check if the classname appears in any other java file
  # We use grep and exclude the file itself.
  # We look for the classname.
  count=$(rg -w "$classname" -t java -g "!$f" | wc -l)
  
  if [ "$count" -eq 0 ]; then
    unused+=("$f")
  fi
done

echo "Potential unused classes:"
for u in "${unused[@]}"; do
  echo "$u"
done
