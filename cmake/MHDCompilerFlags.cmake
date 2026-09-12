# Shared high-performance compile flags (LLVM/Clang preferred).
# Applied to all MHD targets via mhd_apply_flags(<target>).

function(mhd_apply_flags target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "mhd_apply_flags: target '${target_name}' does not exist")
    endif()

    target_compile_features(${target_name} PRIVATE cxx_std_20)

    # Warnings
    target_compile_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )

    # Release / RelWithDebInfo optimizations (Clang/LLVM and GCC)
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
            -O3
            -DNDEBUG
            -fno-math-errno
            -fno-trapping-math
            -ffp-contract=fast
            -fvectorize
            -fslp-vectorize
        >
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
            -O3
            -DNDEBUG
            -fno-math-errno
            -fno-trapping-math
            -ffp-contract=fast
        >
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
            /O2 /DNDEBUG
        >
    )

    # Native CPU tuning when requested (default ON for local builds)
    option(MHD_NATIVE_ARCH "Enable -march=native / -mcpu=native" ON)
    if(MHD_NATIVE_ARCH)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang,GNU>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-march=native>
        )
    endif()

    # Thin LTO for Clang/LLVM Release (optional; can slow CI links)
    option(MHD_ENABLE_LTO "Enable thin LTO for Clang Release builds" OFF)
    if(MHD_ENABLE_LTO)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang>,$<CONFIG:Release>>:-flto=thin>
        )
        target_link_options(${target_name} PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang>,$<CONFIG:Release>>:-flto=thin>
        )
    endif()
endfunction()
