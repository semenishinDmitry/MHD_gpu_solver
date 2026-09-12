## Summary
<!-- What does this PR change and why? -->

## Checklist
- [ ] Tests added/updated (`ctest` passes locally)
- [ ] Verification impact considered (ICs, refs, tolerances)
- [ ] Documentation updated (`docs/`, README) if behaviour/API changed
- [ ] Benchmark impact considered (optional; do not fail on noise)
- [ ] Formatting passes (`./scripts/format.sh check`)
- [ ] Static analysis considered (`cmake --build … --target lint` if clang-tidy available)

## Test plan
- [ ] `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMHD_NATIVE_ARCH=OFF && cmake --build build -j && ctest --test-dir build --output-on-failure`
- [ ] (If numerical) regenerating refs/goldens documented in the PR
