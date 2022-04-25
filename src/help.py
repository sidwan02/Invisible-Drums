import glob
import os

v = ""
a = glob.glob(os.path.join("../data/custom/JPEGImages", v, "*.jpg"))

print(a)
