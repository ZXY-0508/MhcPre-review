#!/bin/bash
# Builds the contest-signed runner against the freshly built custom op.
# Run from the cloud after `cmake .. && make -j8` has succeeded.
set -e

# 1. Locate the generated aclnn header (build/autogen/aclnn_mhc_pre.h).
HDR=$(find /opt/atomgit -name "aclnn_mhc_pre.h" 2>/dev/null | head -1)
if [ -z "$HDR" ]; then
    echo "ERROR: aclnn_mhc_pre.h not found. Did you run cmake+make first?" >&2
    exit 1
fi
AUTOGEN=$(dirname "$HDR")

# 2. Locate the built custom-op library.
LIB=$(find /opt/atomgit -name "libcust_opapi.so" 2>/dev/null | head -1)
if [ -z "$LIB" ]; then
    echo "ERROR: libcust_opapi.so not found." >&2
    exit 1
fi
LIBDIR=$(dirname "$LIB")

# 3. Locate CANN runtime libs (aarch64 path).
CANN_LIB=$(find /usr/local/Ascend/cann-8.5.0 -name "libascendcl.so" 2>/dev/null | head -1)
if [ -z "$CANN_LIB" ]; then
    echo "ERROR: libascendcl.so not found under /usr/local/Ascend." >&2
    exit 1
fi
CANN_LIBDIR=$(dirname "$CANN_LIB")

# 4. Compile with --copy-dt-needed-entries to handle transitive deps.
INCS="-I$AUTOGEN -I$(dirname $CANN_LIBDIR)/include"
LIBS="-L$LIBDIR -lcust_opapi -L$CANN_LIBDIR -lascendcl -lnnopbase"
RPATH="-Wl,-rpath,$LIBDIR -Wl,-rpath,$CANN_LIBDIR"

g++ -std=c++17 -O2 -o runner runner.cpp \
    -Wl,--copy-dt-needed-entries \
    $INCS $LIBS $RPATH
echo "runner built OK -> ./runner"