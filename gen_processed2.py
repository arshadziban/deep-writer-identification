import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

WRITER = "221311076"
SAMPLE = "1.jpeg"
SRC = f"dataset/{WRITER}/{SAMPLE}"

img_bgr = cv2.imread(SRC)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def apply_laplacian(g): return cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F))
def apply_gabor(g):
    responses = [cv2.filter2D(g, cv2.CV_32F,
        cv2.getGaborKernel((21,21), 5.0, t, 10.0, 0.5, 0, ktype=cv2.CV_32F))
        for t in (0, np.pi/4, np.pi/2, 3*np.pi/4)]
    return cv2.convertScaleAbs(np.max(np.stack(responses), axis=0))
def apply_morph_opening(g): return cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
def apply_morph_gradient(g): return cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
def apply_skeleton(g): return (skeletonize(g < 245).astype(np.uint8) * 255)

images = [
    (gray, "Original"),
    (apply_laplacian(gray), "Laplacian\n(Edge Sharpness)"),
    (apply_gabor(gray), "Gabor\n(Stroke Orientation)"),
    (apply_morph_opening(gray), "Morph Opening\n(Noise Removal)"),
    (apply_morph_gradient(gray), "Morph Gradient\n(Stroke Boundary)"),
    (apply_skeleton(gray), "Skeleton\n(Stroke Thinning)"),
]

fig, axes = plt.subplots(1, 6, figsize=(22, 5))
fig.suptitle(
    f"Processed Image Examples — 5 Filters Applied to Original\n(Writer: {WRITER}, Sample: {SAMPLE})",
    fontsize=13, fontweight="bold"
)

for ax, (img, title) in zip(axes, images):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("processed_examples_2.png", dpi=150, bbox_inches="tight")
print("Saved processed_examples_2.png")
