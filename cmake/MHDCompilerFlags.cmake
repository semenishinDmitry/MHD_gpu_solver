# Shared high-performance compile flags (LLVM/Clang preferred).
# Applied to all MHD targets via mhd_apply_flags(<target>).

function(mhd_apply_flags target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "mhd_apply_flags: target '${target_name}' does not exist")
    endif()

    target_compile_features(${target_name} PRIVATE cxx_std_20)

    # Warnings (strict; do not hide with -Wno-* just to pass CI)
    target_compile_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )

    # Release / RelWithDebInfo optimizations
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

    if(MHD_NATIVE_ARCH)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang,GNU>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-march=native>
        )
    endif()

    if(MHD_ENABLE_LTO)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang>,$<CONFIG:Release>>:-flto=thin>
        )
        target_link_options(${target_name} PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:Clang,AppleClang>,$<CONFIG:Release>>:-flto=thin>
        )
    endif()

    if(MHD_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
            target_compile_options(${target_name} PRIVATE
                -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target_name} PRIVATE
                -fsanitize=address,undefined)
        else()
            message(WARNING "MHD_ENABLE_SANITIZERS is set but unsupported for this compiler")
        endif()
    endif()
endfunction()
