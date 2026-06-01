import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random

WRITER = "221311076"
SAMPLE = "1.jpeg"
SRC = f"dataset/{WRITER}/{SAMPLE}"

random.seed(42)
np.random.seed(42)

pil_img = Image.open(SRC).convert("RGB")
w, h = pil_img.size

rot = pil_img.rotate(random.randint(-30, 30))
shift = random.randint(-w // 5, w // 5)
trans = pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))
arr = np.array(pil_img).astype(np.float64)
noisy = Image.fromarray(np.clip(arr + np.random.normal(0, 25, arr.shape), 0, 255).astype(np.uint8))
sharp = ImageEnhance.Sharpness(pil_img).enhance(random.uniform(0.5, 2.0))

images = [
    (np.array(pil_img.convert("L")), "Original"),
    (np.array(rot.convert("L")), f"Rotation\n(±30° Random)"),
    (np.array(trans.convert("L")), f"Translation\n(Horizontal Shift)"),
    (np.array(noisy.convert("L")), "Gaussian Noise\n(σ=25)"),
    (np.array(sharp.convert("L")), "Sharpness\n(0.5×–2.0× Random)"),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle(
    f"Augmented Image Examples — 4 Augmentation Variants\n(Writer: {WRITER}, Sample: {SAMPLE})",
    fontsize=13, fontweight="bold"
)

for ax, (img, title) in zip(axes, images):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("augmented_examples_2.png", dpi=150, bbox_inches="tight")
print("Saved augmented_examples_2.png")
