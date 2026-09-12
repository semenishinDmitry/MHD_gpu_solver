# MHD Verification Reference Values
#
# Generated with:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMHD_NATIVE_ARCH=OFF
#   cmake --build build -j
#   ./build/mhd_verify --dump-refs tests/verification/reference
#
# These are scalar diagnostics (not full field dumps) for FAST CI problems.
# Regenerating after intentional numerical changes is expected; do not tweak
# values merely to silence CI without understanding the change.
#
# Format: MHD_REF 1
#         <key> <value>
