import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
size, grid_size =np.loadtxt('grid.txt')
print(f"size = {size}, grid size= {grid_size}")
x2_energy=np.transpose(np.loadtxt('island.txt'))
g= 1

fig, ax = plt.subplots()

ax.scatter(x2_energy[0],x2_energy[1])
ax.set_title(f"Anharmonic  k={size}, g={g}, grid size = {grid_size}.png")
#ax.set(ylim=(1.36, 1.43), xlim=(0.29, 0.32))
ax.set_xlabel('$< x^2 > $')
ax.set_ylabel('Energy')
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
#plt.show()
plt.savefig(f"plots/C++ Data vtemp, k={size}, g={g}, grid size = {grid_size}.png")

