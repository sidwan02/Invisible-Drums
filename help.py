import numpy as np
import json
from collections import defaultdict

a = defaultdict(int)

# a = np.array([{"hi": 1}, 2, 3, 4]).tolist()

with open("thing.json", "w") as f:
    json.dump(a, f)
