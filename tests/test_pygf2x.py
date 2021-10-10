#!/usr/bin/python3

import unittest
import random
from random import randint,uniform

import gint
from gint import gint as gi

import pygf2x_generic as gf2

random.seed(1234567890)

too_large = int(gi.MAX())+1

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
            
    def test_oversize(self):
        with self.assertRaises(ValueError):
            gf2.sqr(too_large)
            
    def test_overflow(self):
        with self.assertRaises(OverflowError):
            gf2.sqr(1<<(too_large.bit_length()//2+1))
            
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
            
    def test_oversize(self):
        with self.assertRaises(ValueError):
            gf2.mul(too_large,2)
        with self.assertRaises(ValueError):
            gf2.mul(2,too_large)
            
    def test_overflow(self):
        len1 = 1000
        len2 = too_large.bit_length() - len1 +1
        with self.assertRaises(OverflowError):
            gf2.mul((1<<len1),(1<<len2))
            
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


class test_inv(unittest.TestCase):
    @staticmethod
    def model_inv(d, ne):
        nd = d.bit_length()
        e = d
        if ne > nd:
            e <<= ne-nd
        else:
            e >>= nd-ne
        ibits = 1
        while ibits < ne-1:
            ibits = min(ibits<<1, ne-1)
            ei = e >> (ne-ibits)
            e = ei*ei*d
            shft = ne - (2*ibits+nd-2)
            if shft > 0:
                e <<= shft
            else:
                e >>= -shft
        return e

    def test_type(self):
        with self.assertRaises(TypeError):
            gf2.inv(3.14,1)
        with self.assertRaises(TypeError):
            gf2.divmod(0.0,1)
        with self.assertRaises(TypeError):
            gf2.divmod(1,1.5)
    
    def test_0(self):
        with self.assertRaises(ZeroDivisionError):
            gf2.inv(0,1)
        with self.assertRaises(ValueError):
            gf2.inv(1,0)
        with self.assertRaises(ValueError):
            gf2.inv(1,-1)
        with self.assertRaises(ValueError):
            gf2.inv(-1,1)

    def test_1(self):
        self.assertEqual(gf2.inv(1,1),1)

    def test_oversize(self):
        with self.assertRaises(ValueError):
            gf2.inv(too_large, 10)
        with self.assertRaises(OverflowError):
            gf2.inv(10, too_large.bit_length())
            
    def test_small(self):
        for i in range(1,1024):
            i = gi(i)
            ni = i.bit_length()
            self.assertEqual((gi(gf2.inv(i,ni))*i)>>(ni-1), 1<<(ni-1))
            self.assertEqual(gf2.inv(i,ni),self.model_inv(i,ni))

    def test_model(self):
        for i in range(1,10):
            i = gi(randint(1<<(i-1),(1<<i)-1))
            ni = i.bit_length()
            ne = randint(1,10)
            self.assertEqual((gi(self.model_inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
        
    def test_small_coarse(self):
        for i in range(1,1024):
            i = gi(i)
            ni = i.bit_length()
            ne = max(ni-1,1)
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            ne = max(ni//2,1)
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            ne = 1
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))

    def test_small_fine(self):
        for i in range(1,1024):
            i = gi(i)
            ni = i.bit_length()
            ne = ni+1
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            ne = ni*2
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            ne = ni*5+3
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))

    def test_big(self):
        for i in range(1,1024):
            i = gi(randint(1<<(i-1),(1<<i)-1))
            ni = i.bit_length()
            ne = ni
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            ne = max(ni-1,1)
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            ne = ni+1
            self.assertEqual((gi(gf2.inv(i,ne))*i)>>(ni-1), 1<<(ne-1))
            

class test_div(unittest.TestCase):

    @staticmethod
    def model_rinv(d, ne):
        nd = d.bit_length()
        ne_mask = (1<<ne)-1
        e = d
        e &= ne_mask
        ibits = 1
        while (ibits<<1) < ne-1:
            ibits = min(ibits<<1, ne-1)
            ei = e & ((1<<ibits)-1)
            e = ei*ei*d
            e &= ne_mask

        return e
    
    @staticmethod
    def model_divmod(u, d):
        nd = d.bit_length()
        nu = u.bit_length()
        u = gi(u)
        d = gi(d)
        if nd == 0:
            raise ZeroDivisionError()
        if nu == 0:
            return (gi(0), gi(0))
        if nd == 1:
            return (u, gi(0))
        if nu < nd:
            return (gi(0), u)
        if nu == nd:
            return (gi(1), u^d)
        #
        # We know now that nu>nd>1
        #
        nq = nu - nd +1
        #
        # Choose which inverse accuracy to compute
        # 2 <= ne <= nq+1
        # If ne == nq+1 then the whole division is done in just one step
        # 
        #
        ne = min(nq+1,nd)
        nr = nu
        r = u
        q = gi(0)
        e = test_inv.model_inv(d,ne)
        while nr >= nd + ne:
            #
            # Improve q with ne bits each iteration
            #
            dq = ((r >> (nr-ne)) *e) >> (ne-1)      # |dq| = ne
            nqi = nr - nd - (ne-1)                  # |q_i| - |dq|
            q ^= dq << nqi                          # Shift in place for q
            r ^= (dq*d) << nqi                      # Shift in place for r
            nr -= ne
            assert r.bit_length()<=nr
        #
        # Take last step
        #
        m = nr - nd + 1                             # m = |dq|
        dq = ((r >> (nr-m))*(e >> (ne-m))) >> (m-1) # |dq| = m
        q ^= dq
        r ^= (dq*d)
        nr -= m
        assert nr == nd-1
        assert r.bit_length()<=nr

        return (q,r)

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
            
    def test_oversize(self):
        with self.assertRaises(ValueError):
            gf2.divmod(too_large,2)
        with self.assertRaises(ValueError):
            gf2.divmod(2,too_large)
            
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
