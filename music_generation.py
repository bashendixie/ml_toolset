import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10)
y = 2*x
y1 = -x + 3

#plt.figure()
plt.plot(x, y)
plt.plot(x, y1)
plt.xlim(0, 3)
plt.ylim(0, 3)
# draw axes
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')
plt.show()
plt.close()