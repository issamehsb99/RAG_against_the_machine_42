import math


k = 1.5
b  = 0.75
ld1 = 0.625
ld2 = 1
ld3 = 1.375
f = int(input())
tf = (f *(k+1))/(f+k*ld1)
print(tf)