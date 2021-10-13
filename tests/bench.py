#!/usr/bin/python3

from gint import gint as gi

import random
import math
from time import time,sleep
import threading

class mul_thread (threading.Thread):
    def __init__(self, a, b):
        threading.Thread.__init__(self)
        self.daemon = True
        self.a = a
        self.b = b
        self.stop = False
        
    def run(self):
        a = self.a
        b = self.b
        an = a.bit_length()
        bn = b.bit_length()

        self.count = 0
        t0 = time()
        while True:
            for n in range(0,max(1,10000000//(an*bn))):
                a*b
            self.count += n+1
            if self.stop:
                t1 = time()
                break
        self.dt = t1-t0

class div_thread (threading.Thread):
    def __init__(self, u, d):
        threading.Thread.__init__(self)
        self.daemon = True
        self.u = u
        self.d = d
        self.stop = False
        
    def run(self):
        u = self.u
        d = self.d
        un = u.bit_length()
        dn = d.bit_length()

        self.count = 0
        t0 = time()
        while True:
            for n in range(0,max(1,10000000//(un*dn))):
                divmod(u,d)
            self.count += n+1
            if self.stop:
                t1 = time()
                break
        self.dt = t1-t0

random.seed(1234567890)

karatsuba_exp = math.log(3,2)

un = 10000
u=gi(random.randint(1<<(un-1),(1<<un)-1))
imax = 100
print('='*80)
print('>  DIVMOD un=%d'%un)
print('='*80)
print('%5s %5s %12s %10s %6s'%("qn", "dn", "count", "n_count", "bench"))
for i in range(1,imax):
    if i>10 and i<90 and i%10:
        continue
    if i<=50:
        dn = (i*i*2*un)/(imax*imax)
    else:
        dn = un - (imax-i)*(imax-i)*2*un/(imax*imax)
    dn = int(round(dn))
    u=gi(random.randint(1<<(un-1),(1<<un)-1))
    d=gi(random.randint(1<<(dn-1),(1<<dn)-1))
    th = div_thread(u,d)
    th.start()
    sleep(1)
    th.stop = True
    th.join()
    qn = max(un-dn+1,0)
    val = th.count*(dn*qn)**(.5*karatsuba_exp)/1.e6
    print('%5d %5d %12d %10.3f %6.3f'%(qn, dn, th.count, th.count / th.dt, val))
print('-'*80)


n=0
print('='*80)
print('> MUL')
print('='*80)
print('%5s %12s %10s %6s'%("n", "count", "n_count", "bench"))
while n<25000:
    n += max(5,n//5)
    a=gi(random.randint(1<<(n-1),(1<<n)-1))
    b=gi(random.randint(1<<(n-1),(1<<n)-1))
    th = mul_thread(a,b)
    th.start()
    sleep(1)
    th.stop = True
    th.join()
    val = th.count*n**karatsuba_exp/1.e6
    line = '%5d %12d %10.3f %6.3f'%(n, th.count, th.count / th.dt, val)
    print(line)
print('-'*80)


