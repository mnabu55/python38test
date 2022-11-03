from scipy import misc

ascent_image = misc.ascent()

import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(ascent_image)
plt.show()

import numpy as np

image_transformed = np.copy(ascent_image)

size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]


filter = [[0,1,0], [1,-4,1], [0,1,0]]

weight = 1


