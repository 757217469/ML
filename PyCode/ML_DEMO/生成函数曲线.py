import numpy as np
import math
import matplotlib.pyplot as plt

x1 = np.linspace(0, 1, 1000)
print(x1)
y1 = [math.exp(5 * _ - 4) for _ in x1]
plt.plot(x1, y1)
# y2 = [math.exp(5 * _ - 4) for _ in x1]
# plt.plot(x1, y2)
# y3 = [2 * _ for _ in x1]
# y = [y1[_] + y2[_] for _ in range(len(x1))]
# plt.plot(x1,y)
# plt.plot(x1, y3)
plt.show()
print('                                                                 precision    recall  f1-score   support\n\n                                                        class 0     0.9720    0.9918    0.9818       980\nclass 1class 2class 3class 4class 5class 6class 7class 8class 9     0.9852    0.9947    0.9899      1135\n\n                                                       accuracy                         0.9697     10000\n                                                      macro avg     0.9695    0.9693    0.9694     10000\n                                                   weighted avg     0.9697    0.9697    0.9697     10000\n')
