#!/bin/bash

rm -rf soil.sock run.sh
make clean
make

directory=$(pwd)

cat > "run.sh" <<EOF
cd $directory
./main 1>out.log 2>err.log&
python3 pread.py
EOF
chmod +x run.sh
