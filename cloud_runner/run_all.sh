#!/bin/bash
# Group_vec candidate: one-shot cloud verification (E1/E2/E3).
#
# Prereq: this directory is synced into /opt/atomgit/code (or run from the
# repo root the contest provides, containing code/).
#
# Usage:  bash run_all.sh
set -e
cd "$(dirname "$0")"

echo "=== [0/3] build custom op (cmake + make) ==="
if [ ! -f build/.built ]; then
    mkdir -p build && cd build
    cmake .. > cmake.log 2>&1 || { tail -50 cmake.log; exit 1; }
    make -j8 > make.log 2>&1 || { tail -80 make.log; exit 1; }
    touch .built
    cd ..
fi
echo "build OK"

echo "=== [1/3] build runners ==="
cd cloud_runner
bash build_runner.sh                       # -> ./runner  (E3)
HDR=$(find /opt/atomgit -name "aclnn_mhc_pre.h" 2>/dev/null | head -1)
AUTOGEN=$(dirname "$HDR")
LIB=$(find /opt/atomgit -name "libcust_opapi.so" 2>/dev/null | head -1)
LIBDIR=$(dirname "$LIB")
CANN_LIB=$(find /usr/local/Ascend/cann-8.5.0 -name "libascendcl.so" 2>/dev/null | head -1)
CANN_LIBDIR=$(dirname "$CANN_LIB")
g++ -std=c++17 -O2 -o probe probe_e1e2.cpp \
    -Wl,--copy-dt-needed-entries \
    -I"$AUTOGEN" -I"$(dirname "$CANN_LIBDIR")/include" \
    -L"$LIBDIR" -lcust_opapi -L"$CANN_LIBDIR" -lascendcl -lnnopbase \
    -Wl,-rpath,"$LIBDIR" -Wl,-rpath,"$CANN_LIBDIR"
echo "runners built"

echo "=== [2/3] E1+E2: protocol + relaunch probe ==="
./probe

echo "=== [3/3] E3: full shape matrix (correctness + latency) ==="
./runner

echo "run_all: ALL GREEN"
