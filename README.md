# gint
A Python class for (large) polynomials over GF(2), based on Python's built-in integers, efficiently implemented in C

The purpose of this package is to support polynomial algebra over GF(2). The
polynomial ring is infinite, operations are NOT computed modulo anything.

This package does not depend on numpy nor any other package, instead Pythons
built-in variable-sized integers are used. A Python subclass `gint` is derived
from the built-in `int` where the appropriate 
operators +, -, \*, /, %, divmod are defined. Exponentiation with standard
integer exponent using ** operator is also allowed. Boolean operators &, |, ^
are supported, even mixed with integers (returning gint). Shift operators with
integer shift are allowed. 

Machine instructions for multiplication of GF(2) polynomials are
supported for some CPU models through different assembler variants of core functions.

There is a performance penalty due to the Python integer design being based
on 15- or 30-bit chunks. However even the generic implementation performs
far better than any pure Python implementation, especially for large polynomials.
