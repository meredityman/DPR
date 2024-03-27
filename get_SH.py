import sys
sys.path.append('utils')


from pathlib import Path

import numpy as np
import cv2

from utils_shtools import *
from utils_SH import *

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))
#-----------------------------------------------------------------


input_dir = Path("data/envmaps")

image_paths = input_dir.glob("*.png")

for img_path in image_paths:

    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (1024, 1024))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #img = np.random.rand(180, 360) * np.tile(np.linspace(0.0, 1.0, 360), [180, 1])
    sh = np.array(shtools_getSH(img))
    sh = shtools_matrix2vec(sh)
    sh = sh[0:9]
    sh *= 0.2

    print(sh.shape)

    # rendering half-sphere
    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid


    output_img_path = Path(input_dir, "vis", img_path.name)
    cv2.imwrite(str(output_img_path), shading)

    output_sh_path = Path(input_dir, "sh", f"{img_path.stem}.txt")
    text = "\n".join([ str(s) for s in sh])
    with open(str(output_sh_path), "w") as f:
        f.write(text)