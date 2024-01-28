import rand
import numpy as np
y= [[1,2,3,4,5,6,7,8,9,10,11,12,np.nan]]

y = np.array(y)

y.reshape(1, -1)
print(y)

tetest = [8,9]

ind = ~np.isnan(y)
ind = np.tile(ind, (1, 13))

print(ind)

X = y[1, tetest]
print(X)

for index in tetest:
    print(y[1][index])