#!/usr/bin/python3

import unittest
import random
from random import randint,uniform

import gint
from gint import gint as gi

import pygf2x_generic as gf2

random.seed(1234567890)


class test_sqr(unittest.TestCase):

    @staticmethod
    def model_sqr(x):
        p = 0
        for i in range(0,x.bit_length()):
            if (x>>i)&1:
                p ^= (x<<i)
        return p

    def test_type(self):
        with self.assertRaises(TypeError):
            gf2.sqr(3.14)

    def test_neg(self):
        with self.assertRaises(ValueError):
            gf2.sqr(-10)
            
    def test_0(self):
        self.assertEqual(gf2.sqr(0),0)
        
    def test_1(self):
        self.assertEqual(gf2.sqr(1),1)

    def test_5(self):
        for x in range(0,1<<5):
            self.assertEqual(gf2.sqr(x),self.model_sqr(x))
                
    def test_15(self):
        for n in range(0,100):
            x = randint(1<<5,(1<<15)-1)
            self.assertEqual(gf2.sqr(x),self.model_sqr(x))

    def test_30(self):
        for n in range(0,100):
            x = randint(1<<15,(1<<30)-1)
            self.assertEqual(gf2.sqr(x),self.model_sqr(x))

    def test_60(self):
        for n in range(0,100):
            x = randint(1<<30,(1<<60)-1)
            self.assertEqual(gf2.sqr(x),self.model_sqr(x))

    def test_1000(self):
        for n in range(0,100):
            x = randint(1,(1<<1000)-1)
            self.assertEqual(gf2.sqr(x),self.model_sqr(x))

            
class test_mul(unittest.TestCase):

    @staticmethod
    def model_mul(l,r):
        p = 0
        for i in range(0,l.bit_length()):
            if (l>>i)&1:
                p ^= (r<<i)
        return p

    def test_type(self):
        with self.assertRaises(TypeError):
            gf2.mul(3.14,1)
        with self.assertRaises(TypeError):
            gf2.mul(1,3.14)
        with self.assertRaises(TypeError):
            gf2.mul(3.14,0)

    def test_neg(self):
        with self.assertRaises(ValueError):
            gf2.mul(-10,5)
        with self.assertRaises(ValueError):
            gf2.mul(10,-5)
        with self.assertRaises(ValueError):
            gf2.mul(-10,-5)
            
    def test_0(self):
        self.assertEqual(gf2.mul(0,0),0)
        self.assertEqual(gf2.mul(0,1),0)
        self.assertEqual(gf2.mul(0,(1<<5)-1),0)
        self.assertEqual(gf2.mul(0,(1<<15)-1),0)
        self.assertEqual(gf2.mul(0,(1<<30)-1),0)
        self.assertEqual(gf2.mul(0,(1<<60)-1),0)
        
    def test_1(self):
        self.assertEqual(gf2.mul(10,1),10)
        self.assertEqual(gf2.mul(1,10),10)
        self.assertEqual(gf2.mul((1<<30)-1,1),(1<<30)-1)
        self.assertEqual(gf2.mul(1,(1<<30)-1),(1<<30)-1)
        self.assertEqual(gf2.mul((1<<60)-1,1),(1<<60)-1)
        self.assertEqual(gf2.mul(1,(1<<60)-1),(1<<60)-1)

    def test_5_5(self):
        for l in range(0,1<<5):
            for r in range(0,1<<5):
                self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))
                
    def test_15_5(self):
        for n in range(0,100):
            l = randint(1<<5,(1<<15)-1)
            r = randint(1,(1<<5)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_30_5(self):
        for n in range(0,100):
            l = randint(1<<15,(1<<30)-1)
            r = randint(1,(1<<5)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_60_5(self):
        for n in range(0,100):
            l = randint(1<<30,(1<<60)-1)
            r = randint(1,(1<<5)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_15_15(self):
        for n in range(0,100):
            l = randint(1<<5,(1<<15)-1)
            r = randint(1<<5,(1<<15)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_30_15(self):
        for n in range(0,100):
            l = randint(1<<15,(1<<30)-1)
            r = randint(1<<5,(1<<15)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_60_15(self):
        for n in range(0,100):
            l = randint(1<<30,(1<<60)-1)
            r = randint(1<<5,(1<<15)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_30_30(self):
        for n in range(0,100):
            l = randint(1<<15,(1<<30)-1)
            r = randint(1<<15,(1<<30)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_60_30(self):
        for n in range(0,100):
            l = randint(1<<30,(1<<60)-1)
            r = randint(1<<15,(1<<30)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_60_60(self):
        for n in range(0,100):
            l = randint(1<<30,(1<<60)-1)
            r = randint(1<<30,(1<<60)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))

    def test_1000_1000(self):
        for n in range(0,100):
            l = randint(1,(1<<1000)-1)
            r = randint(1,(1<<1000)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))



class test_div(unittest.TestCase):

    @staticmethod
    def model_inv(d, ne):
        nd = d.bit_length()
        e = d
        if ne > nd:
            e <<= ne-nd
        else:
            e >>= nd-ne
        ibits = 1
        while (ibits<<1) < ne-1:
            ibits = min(ibits<<1, ne-1)
            ei = e >> (ne-ibits)
            e = ei*ei*d
            shft = ne - (2*ibits+nd-2)
            if shft > 0:
                e <<= shft
            else:
                e >>= -shft
        return e

    @staticmethod
    def model_div(u,d):
        # TODO: untested!
        nu = u.bit_length()
        nd = d.bit_length()
        ne = nu-nd+2
        e = model_inv(d,ne)
        q = (u*e) >> (ne+nd-2)
        r = u-q*d
        return q,r

    def test_type(self):
        with self.assertRaises(TypeError):
            gf2.divmod(3.14,1)
        with self.assertRaises(TypeError):
            gf2.divmod(1,3.14)
        with self.assertRaises(TypeError):
            gf2.divmod(3.14,0)
    
    def test_0_0(self):
        with self.assertRaises(ZeroDivisionError):
            gf2.divmod(0,0)

    def test_inv_0(self):
        with self.assertRaises(ZeroDivisionError):
            gf2.divmod(1,0)
        with self.assertRaises(ZeroDivisionError):
            gf2.divmod(0x1a,0)
        with self.assertRaises(ZeroDivisionError):
            gf2.divmod(1<<30,0)

    def test_neg(self):
        with self.assertRaises(ValueError):
            gf2.divmod(-10,5)
        with self.assertRaises(ValueError):
            gf2.divmod(10,-5)
        with self.assertRaises(ValueError):
            gf2.divmod(-10,-5)
            
    def test_0(self):
        self.assertEqual(gf2.divmod(0,1),(0,0))
        self.assertEqual(gf2.divmod(0,(1<<5)-1),(0,0))
        self.assertEqual(gf2.divmod(0,(1<<15)-1),(0,0))
        self.assertEqual(gf2.divmod(0,(1<<30)-1),(0,0))
        self.assertEqual(gf2.divmod(0,(1<<60)-1),(0,0))
        
    def test_1(self):
        self.assertEqual(gf2.divmod(10,1),(10,0))
        self.assertEqual(gf2.divmod(1<<30,1),(1<<30,0))
        self.assertEqual(gf2.divmod((1<<60)-1,1),((1<<60)-1,0))

    def test_5_5(self):
        for u in range(0,1<<5):
            for d in range(1,1<<5):
                q,r = gf2.divmod(u,d)
                self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
                self.assertTrue(r.bit_length() < d.bit_length(), 'u=%x d=%x'%(u,d))
                
    def test_15_5(self):
        for n in range(0,100):
            u = randint(1<<5,(1<<15)-1)
            d = randint(1,(1<<5)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_30_5(self):
        for n in range(0,100):
            u = randint(1<<15,(1<<30)-1)
            d = randint(1,(1<<5)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_60_5(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1,(1<<5)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_15_15(self):
        for n in range(0,100):
            u = randint(1<<5,(1<<15)-1)
            d = randint(1<<5,(1<<15)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_30_15(self):
        for n in range(0,100):
            u = randint(1<<15,(1<<30)-1)
            d = randint(1<<5,(1<<15)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_60_15(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1<<5,(1<<15)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_30_30(self):
        for n in range(0,100):
            u = randint(1<<15,(1<<30)-1)
            d = randint(1<<15,(1<<30)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_60_30(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1<<15,(1<<30)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_60_60(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1<<30,(1<<60)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length(), 'u=%x d=%x'%(u,d))
            
    def test_10000_100(self):
        for n in range(0,100):
            u = randint(0,(1<<10000)-1)
            d = randint(1,(1<<100)-1)
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length(),'divmod(\n%x,\n%x)\n%d,%d\n%x'%(u,d,r.bit_length(),d.bit_length(),r))
            
    def test_10000_10000(self):
        for n in range(0,100):
            lu = uniform(0,10)
            u = int(10**lu)
            d = int(10**uniform(0,lu))
            q,r = gf2.divmod(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())


if __name__ == '__main__':
    unittest.main()
