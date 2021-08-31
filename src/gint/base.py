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

class gint_ring:
    def __init__(self, n):
        assert isinstance(n, int)
        assert n==0 or n&1==1
        self.n = n
        self.r = bit_length(n)

class gint_ring_2p(gint_ring):
    ''' The reducing polynomial is a power of 2 '''
    def __init__(self, n):
        super().__init__(n)

class gint_ring_no2(gint_ring):
    ''' The reducing polynomial has no common divisor with 2 '''
    def __init__(self, n):
        super().__init__(n)


class gint(int):
    def __init__(self, i0=0, ring=None):
        ''' Galois polynomial, optionally element of ring (modulo n) '''
        int.__init__(i0)
        self.ring = ring
    
    def __mul__(self,rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot multiply gint with %s'%type(rhs))
        return gint(pygf2x.mul(self,rhs))

    def __rmul__(self,lhs):
        if not isinstance(lhs, gint):
            raise TypeError('Cannot multiply %s with gint'%type(lhs))
        return gint(pygf2x.mul(lhs,self))
    
    def __div__(self,rhs):
        if not isinstance(rhs, gint):
            raise TypeError('Cannot divide gint with %s'%type(rhs))
        return gint(pygf2x.div(self,rhs)[0])

    def __rdiv__(self,lhs):
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


def main():
    import random
    n=gint(random.randint(0,10**213))
    d=gint(random.randint(0,10**29))
    fmt='%-20s=%-30X'
    print(fmt%('n',n))
    print(fmt%('d',d))

    #q,r = pygf2x.div(n,d)
    #q=gint(q)
    #r=gint(r)
    #print(fmt%('q',q))
    #print(fmt%('r',r))

    q,r = divmod(n,d)
    print(fmt%('q',q))
    print(fmt%('r',r))

    #p = pygf2x.mul(q,d)
    #print(fmt%('q*d',p))
    p = q*d
    print(fmt%('q*d',p))
    #print(fmt%('q*d+r',p^r))
    print(fmt%('q*d+r',p+r))

    print(p+r == n)

if __name__ == '__main__':
    main()
