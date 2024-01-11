import rand
import numpy as np
y= [[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,8,9,10,11,12]]

tetest = [8,9]

y = np.array(y)
X = y[1, tetest]
print(X)

for index in tetest:
    print(y[1][index])