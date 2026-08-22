import os
from PIL import Image

src_image_path = r"D:\DimensionToTsuLovers_cn{A25922C8-2EC5-4D92-9AE3-73A2C0D4665B}\豪華版特典\ディメンション凸ラバース!!壁紙\PC\3840×2160 px\3840x2160px_D_suisen.png"
target_dir = r"D:\Blog_file\astro-public\images\hero"

os.makedirs(target_dir, exist_ok=True)

print(f"Loading source image from: {src_image_path}")
img = Image.open(src_image_path)
orig_w, orig_h = img.size
print(f"Original image size: {orig_w}x{orig_h}")

# Save original copy
orig_copy_path = os.path.join(target_dir, "suisen-hero-original.png")
if not os.path.exists(orig_copy_path):
    img.save(orig_copy_path, optimize=True)
    print(f"Saved: {orig_copy_path} ({os.path.getsize(orig_copy_path) / 1024 / 1024:.2f} MB)")

# Generate 3840w WebP
p3840_webp = os.path.join(target_dir, "suisen-hero-3840w.webp")
img.save(p3840_webp, "WEBP", quality=90, method=6)
print(f"Saved: {p3840_webp} ({os.path.getsize(p3840_webp) / 1024 / 1024:.2f} MB)")

# Generate 1440w WebP and PNG
h1440 = int(orig_h * (1440 / orig_w))
img_1440 = img.resize((1440, h1440), Image.Resampling.LANCZOS)

p1440_webp = os.path.join(target_dir, "suisen-hero-1440w.webp")
img_1440.save(p1440_webp, "WEBP", quality=88, method=6)
print(f"Saved: {p1440_webp} ({os.path.getsize(p1440_webp) / 1024:.1f} KB)")

p1440_png = os.path.join(target_dir, "suisen-hero-1440w.png")
img_1440.save(p1440_png, "PNG", optimize=True)
print(f"Saved: {p1440_png} ({os.path.getsize(p1440_png) / 1024 / 1024:.2f} MB)")

# Generate 750w WebP (Mobile)
h750 = int(orig_h * (750 / orig_w))
img_750 = img.resize((750, h750), Image.Resampling.LANCZOS)

p750_webp = os.path.join(target_dir, "suisen-hero-750w.webp")
img_750.save(p750_webp, "WEBP", quality=85, method=6)
print(f"Saved: {p750_webp} ({os.path.getsize(p750_webp) / 1024:.1f} KB)")

p750_png = os.path.join(target_dir, "suisen-hero-750w.png")
img_750.save(p750_png, "PNG", optimize=True)
print(f"Saved: {p750_png} ({os.path.getsize(p750_png) / 1024:.1f} KB)")

# Also create suisen-hero.webp alias for convenient fallback
default_webp = os.path.join(target_dir, "suisen-hero.webp")
img_1440.save(default_webp, "WEBP", quality=88, method=6)

print("All hero image variants successfully generated.")
