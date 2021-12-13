#!/usr/bin/python3
################################################################################
#
# Copyright (c) 2020 Oskar Enoksson. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Description:
#
# Define gint class (inifinite polynomial field over GF(2))
#
################################################################################

import pygf2x

class gint(int):
    ''' Unbounded polynomial over GF(2)
    This class implements unlimited polynomials over GF(2).
    It is derived from the built-in `int` object.
    Overloaded operators for algebra include +,-,*,/,%,divmod.
    Exponentiation with integer exponent using ** operator is supported.
    Boolean operators &,|,^ are supported, even with integers (returning gint)
    Shift operators with integer shift is allowed.
    '''
    def __init__(self, i0=0):
        ''' Create from integer '''
        if i0.bit_length() > self.get_MAX_BITS():
            raise OverflowError("Attempt to create a gint with value out of range")
        int.__init__(i0)
    
    def __mul__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot multiply gint with %s'%type(value).__name__)
        return gint(pygf2x.mul(self,value))

    def __rmul__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot multiply %s with gint'%type(value).__name__)
        return gint(pygf2x.mul(value,self))
    
    def __truediv__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot divide gint with %s'%type(value).__name__)
        return gint(pygf2x.divmod(self,value)[0])

    def __rtruediv__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot divide %s with gint'%type(value).__name__)
        return gint(pygf2x.divmod(self,value)[0])

    def __floordiv__(self,value):
        raise TypeError("Don't use // with gint")

    def __rfloordiv__(self,value):
        raise TypeError("Don't use // with gint")

    def __divmod__(self, value):
        if not isinstance(value, gint):
            raise TypeError('Cannot divmod gint with %s'%type(value).__name__)
        return tuple(map(lambda x : gint(x), pygf2x.divmod(self,value)))

    def __rdivmod__(self, value):
        if not isinstance(value, gint):
            raise TypeError('Cannot divmod %s with gint'%type(value).__name__)
        return tuple(map(lambda x : gint(x), pygf2x.divmod(value,self)))

    def __mod__(self, value):
        if not isinstance(value, gint):
            raise TypeError('Cannot modulo gint with %s'%type(value).__name__)
        return gint(pygf2x.divmod(self,value)[1])

    def __rmod__(self, value):
        if not isinstance(value, gint):
            raise TypeError('Cannot modulo %s with gint'%type(value).__name__)
        return gint(pygf2x.divmod(value,self)[1])

    def __add__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot add gint with %s'%type(value).__name__)
        return gint(int.__xor__(self,value))
    
    def __radd__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot add %s with gint'%type(value).__name__)
        return gint(int.__xor__(value,self))

    def __sub__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot subtract %s from gint'%type(value).__name__)
        return gint(int.__xor__(self,value))
    
    def __rsub__(self,value):
        if not isinstance(value, gint):
            raise TypeError('Cannot subtract gint from %s'%type(value).__name__)
        return gint(int.__xor__(value,self))

    def __neg__(self):
        return gint(self)

    def __pos__(self):
        return gint(self)
    
    def __invert__(self):
        raise TypeError("Cannot invert gint")
    
    def __lshift__(self,shift):
        return gint(int.__lshift__(self,shift))

    def __rshift__(self,shift):
        return gint(int.__rshift__(self,shift))

    def __or__(self,value):
        return gint(int.__or__(self,value))

    def __ror__(self,value):
        return gint(int.__ror__(self,value))

    def __and__(self,value):
        return gint(int.__and__(self,value))

    def __rand__(self,value):
        return gint(int.__rand__(self,value))

    def __xor__(self,value):
        return gint(int.__xor__(self,value))
    
    def __rxor__(self,value):
        return gint(int.__rxor__(self,value))

    def __rrshift__(self,value):
        raise TypeError("Don't use gint to right-shift")

    def __rlshift__(self,value):
        raise TypeError("Don't use gint to left-shift")
    
    def __index__(self,value):
        raise TypeError("Don't use gint as index")

    def __abs__(self):
        raise TypeError("Abs of gint doesn't make sense")

    def conjugate(self):
        raise TypeError("Conjugate of gint doesn't make sense")

    def __pow__(self, value):
        if not type(value) is int or value<0:
            raise TypeError("gint must only be exponentiated with a non-negative integer")
        if value==0:
            if not self:
                raise ValueError("gint(0)^0 is undefined")
            return gint(1)

        if value==1:
            return gint(self)

        if value*self.bit_length() > self.get_MAX_BITS():
            raise OverflowError("Exponentiation result out of range")
        
        result = gint(1)
        prod = gint(self)
        if value & 1:
            result *= prod
        value >>= 1
        while value:
            prod = gint(pygf2x.sqr(prod))
            if value & 1:
                result *= prod
            value >>= 1
        return result

    def __rpow__(self,value):
        raise TypeError("gint as exponent doesn't make sense")

    def inv(self, nbits):
        ''' Multiplicative inverse of x, with nbits precision, i.e. 
        x*inv(x) = (1<<(x.bit_length()+nbits-2)) + r, where r.bit_length() < nbits
        '''
        return gint(pygf2x.inv(self, nbits))

    def rinv(self, nbits):
        ''' Multiplicative inverse of x, with nbits precision, i.e. 
        x*inv(x) = (r<<(nbits-1)) + 1, where r.bit_length() < nbits
        '''
        return gint(pygf2x.rinv(self, nbits))

    
    @classmethod
    def from_bytes(cls,value):
        return gint(int.from_bytes(value))

    @classmethod
    def get_MAX_BITS(cls):
        return pygf2x.get_MAX_BITS()

    @classmethod
    def set_MAX_BITS(cls, nbits):
        return pygf2x.set_MAX_BITS(nbits)
