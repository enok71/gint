#!/usr/bin/python3
################################################################################
#
# Copyright (c) 2020 Oskar Enoksson. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Description:
#
# Define gint class (inifinite polynomial field) and gint_ring classes (polynomials
# modulo some constant)
#
################################################################################

import pygf2x_generic as pygf2x

class gint(int):
    def __init__(self, i0=0):
        ''' Unbounded polynomial over GF(2) '''
        int.__init__(i0)
    
    def __mul__(self,rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot multiply gint with %s'%type(rhs))
        return gint(pygf2x.mul(self,rhs))

    def __rmul__(self,lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot multiply %s with gint'%type(lhs))
        return gint(pygf2x.mul(lhs,self))
    
    def __truediv__(self,rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot divide gint with %s'%type(rhs))
        return gint(pygf2x.div(self,rhs)[0])

    def __rtruediv__(self,lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot divide %s with gint'%type(lhs))
        return gint(pygf2x.div(self,rhs)[0])

    def __divmod__(self, rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot divmod gint with %s'%type(rhs))
        return tuple(map(lambda x : gint(x), pygf2x.div(self,rhs)))

    def __rdivmod__(self, lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot divmod %s with gint'%type(lhs))
        return tuple(map(lambda x : gint(x), pygf2x.div(lhs,self)))

    def __mod__(self, rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot modulo gint with %s'%type(rhs))
        return gint(pygf2x.div(self,rhs)[1])

    def __rmod__(self, lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot modulo %s with gint'%type(lhs))
        return gint(pygf2x.div(lhs,self)[1])

    def __add__(self,rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot add gint with %s'%type(rhs))
        return gint(int.__xor__(self,rhs))
    
    def __radd__(self,lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot add %s with gint'%type(lhs))
        return gint(int.__xor__(lhs,self))

    def __sub__(self,rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot add gint with %s'%type(rhs))
        return gint(int.__xor__(self,rhs))
    
    def __rsub__(self,lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot add %s with gint'%type(lhs))
        return gint(int.__xor__(lhs,self))

    def __lshift__(self,shift):
        return gint(int.__lshift__(self,shift))

    def __rshift__(self,shift):
        return gint(int.__rshift__(self,shift))

def main():
    # Test
    import random
    n=gint(random.randint(0,10**213))
    d=gint(random.randint(0,10**29))
    fmt='%-20s=%-30X'
    print(fmt%('n',n))
    print(fmt%('d',d))

    q,r = divmod(n,d)
    print(fmt%('q',q))
    print(fmt%('r',r))

    p = q*d
    print(fmt%('q*d',p))
    print(fmt%('q*d+r',p+r))

    print(p+r == n)

if __name__ == '__main__':
    main()
