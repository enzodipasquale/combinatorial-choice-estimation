import numpy as np
# print((np.ones((493,493,3)) @ np.ones(3)).shape)
for i in range(300,350):
    with np.errstate(divide='ignore', overflow='ignore', invalid='ignore'):
        result = np.ones((i, 3)) @ np.ones(3)
        print(result.shape)
# print((np.ones(3) @ np.ones(3)).shape)

