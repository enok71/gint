################################################################################
#
# Copyright (c) 2020 Oskar Enoksson. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Description:
# Run this script to build the Python package contained in this directory
#
################################################################################

from setuptools import setup, Extension

setup_args = dict(
    ext_modules = [
        Extension(
            'pygf2x_generic',
            ['c_ext/pygf2x_generic.c'],
            include_dirs = ['c_ext'],
            #extra_compile_args = ["-march=native"],
            py_limited_api = True
        )
    ]
)

setup(**setup_args)

