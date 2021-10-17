#!/usr/bin/python3

import unittest
import random
from random import randint,uniform

import gint
from gint import gint as gi

import pygf2x_generic as gf2

random.seed(1234567890)


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

    def test_allbitlengths(self):
        for nl in range(1,100):
            for nr in range(1,100):
                l = randint(1<<(nl-1), (1<<nl)-1)
                r = randint(1<<(nr-1), (1<<nr)-1)
                self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))
    
    def test_1000_1000(self):
        for n in range(0,100):
            l = randint(1,(1<<1000)-1)
            r = randint(1,(1<<1000)-1)
            self.assertEqual(gf2.mul(l,r),self.model_mul(l,r))


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
            gf2.div(3.14,1)
        with self.assertRaises(TypeError):
            gf2.div(1,3.14)
        with self.assertRaises(TypeError):
            gf2.div(3.14,0)
    
    def test_0_0(self):
        with self.assertRaises(ZeroDivisionError):
            gf2.div(0,0)

    def test_inv_0(self):
        with self.assertRaises(ZeroDivisionError):
            gf2.div(1,0)
        with self.assertRaises(ZeroDivisionError):
            gf2.div(0x1a,0)
        with self.assertRaises(ZeroDivisionError):
            gf2.div(1<<30,0)

    def test_neg(self):
        with self.assertRaises(ValueError):
            gf2.div(-10,5)
        with self.assertRaises(ValueError):
            gf2.div(10,-5)
        with self.assertRaises(ValueError):
            gf2.div(-10,-5)
            
    def test_0(self):
        self.assertEqual(gf2.div(0,1),(0,0))
        self.assertEqual(gf2.div(0,(1<<5)-1),(0,0))
        self.assertEqual(gf2.div(0,(1<<15)-1),(0,0))
        self.assertEqual(gf2.div(0,(1<<30)-1),(0,0))
        self.assertEqual(gf2.div(0,(1<<60)-1),(0,0))
        
    def test_1(self):
        self.assertEqual(gf2.div(10,1),(10,0))
        self.assertEqual(gf2.div(1<<30,1),(1<<30,0))
        self.assertEqual(gf2.div((1<<60)-1,1),((1<<60)-1,0))

    def test_5_5(self):
        for u in range(0,1<<5):
            for d in range(1,1<<5):
                q,r = gf2.div(u,d)
                self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
                self.assertTrue(r.bit_length() < d.bit_length(), 'u=%x d=%x'%(u,d))
                
    def test_15_5(self):
        for n in range(0,100):
            u = randint(1<<5,(1<<15)-1)
            d = randint(1,(1<<5)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_30_5(self):
        for n in range(0,100):
            u = randint(1<<15,(1<<30)-1)
            d = randint(1,(1<<5)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_60_5(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1,(1<<5)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_15_15(self):
        for n in range(0,100):
            u = randint(1<<5,(1<<15)-1)
            d = randint(1<<5,(1<<15)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_30_15(self):
        for n in range(0,100):
            u = randint(1<<15,(1<<30)-1)
            d = randint(1<<5,(1<<15)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_60_15(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1<<5,(1<<15)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_30_30(self):
        for n in range(0,100):
            u = randint(1<<15,(1<<30)-1)
            d = randint(1<<15,(1<<30)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())
            
    def test_60_30(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1<<15,(1<<30)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())

    def test_60_60(self):
        for n in range(0,100):
            u = randint(1<<30,(1<<60)-1)
            d = randint(1<<30,(1<<60)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length(), 'u=%x d=%x'%(u,d))
            
    def test_allbitlengths(self):
        for nd in range(1,100):
            for nu in range(1,100):
                u = randint(1<<(nu-1), (1<<nu)-1)
                d = randint(1<<(nd-1), (1<<nd)-1)
                q,r = gf2.div(u,d)
                self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
                self.assertTrue(r.bit_length() < d.bit_length(),'divmod(\n%x,\n%x)\n%d,%d\n%x'%(u,d,r.bit_length(),d.bit_length(),r))
            
    def test_10000_100(self):
        for n in range(0,100):
            u = randint(0,(1<<10000)-1)
            d = randint(1,(1<<100)-1)
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length(),'divmod(\n%x,\n%x)\n%d,%d\n%x'%(u,d,r.bit_length(),d.bit_length(),r))
            
    def test_10000_10000(self):
        for n in range(0,100):
            lu = uniform(0,10)
            u = int(10**lu)
            d = int(10**uniform(0,lu))
            q,r = gf2.div(u,d)
            self.assertEqual(gf2.mul(q,d)^r,u,'divmod(%x,%x)'%(u,d))
            self.assertTrue(r.bit_length() < d.bit_length())


if __name__ == '__main__':
    unittest.main()
