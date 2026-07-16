"""C/C++ target wrappers for applications built against HRX."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

_HRX_RUNTIME_DEFINES = "@hrx_system//runtime/src:defines"


def _runtime_deps(deps):
    return (deps or []) + [_HRX_RUNTIME_DEFINES]


def iree_runtime_cc_library(name, deps = None, **kwargs):
    cc_library(
        name = name,
        deps = _runtime_deps(deps),
        **kwargs
    )


def iree_runtime_cc_binary(name, deps = None, **kwargs):
    cc_binary(
        name = name,
        deps = _runtime_deps(deps),
        linkstatic = True,
        **kwargs
    )


def iree_runtime_cc_test(name, deps = None, **kwargs):
    cc_test(
        name = name,
        deps = _runtime_deps(deps),
        linkstatic = True,
        **kwargs
    )


def iree_runtime_cc_benchmark(name, deps = None, **kwargs):
    cc_binary(
        name = name,
        deps = _runtime_deps(deps),
        linkstatic = True,
        testonly = True,
        **kwargs
    )
