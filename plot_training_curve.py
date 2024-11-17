import matplotlib.pyplot as plt
import re
import numpy as np

with open("./b64/msanet-k4-lr-4e-3_log.txt", mode='r', encoding='utf-8') as w:
    acc = w.readlines()

p = [float(re.findall(r"Prec@1 ([0-9\.]+)", x)[0]) for x in acc]
win = 5
avg_p = [
    np.mean(p[max(0, i - win):min(len(p), i + win)]) for i in range(len(p))
]
plt.ylim([85, 90])
plt.grid()
plt.plot(p)
plt.plot(avg_p, 'orange')
plt.savefig("curve.png")
