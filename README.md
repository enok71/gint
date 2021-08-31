# gint
A Python class for polynomials over GF(2), based on Python's built-in integers, efficiently implemented in C

The purpose of this package is to support Polynomial algebra over GF(2). This
includes finite fields (polynomials modulo some constant) as well as unlimited
polynomials.

The package does not depend on numpy nor any other non-trivial package, instead Pythons built-in variable-sized integers are
used. A Python subclass derived from the built-in `int` is included where the appropriate
operators are overloaded. Using Python integers instead of numpy eliminates
the dependency on numpy and it's list of dependencies.

The performance has a small penalty due to the Python integer design based
on 15- or 30-bit chunks. However the generic implementation in C still performs
way better than any pure Python implementation.

The plan is to extended with customized machine specific implementations
that e.g. take advantage of modern CPU instructions for Galois multiplication
in order to improve performance further.
