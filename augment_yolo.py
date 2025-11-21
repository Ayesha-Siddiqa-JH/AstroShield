# augment_yolo.py
# Offline augmentation for YOLO (writes augmented images and labels to train_aug)
import os, glob, cv2
import albumentations as A

ROOT = r"D:\My Downloads\AstroShield_project"
train_images = os.path.join(ROOT, "data", "raw", "train", "images")
train_labels = os.path.join(ROOT, "data", "raw", "train", "labels")
OUT_IMAGES = os.path.join(ROOT, "data", "raw", "train_aug", "images")
OUT_LABELS = os.path.join(ROOT, "data", "raw", "train_aug", "labels")
os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Rotate(limit=15, p=0.4),
    A.Cutout(num_holes=1, max_h_size=50, max_w_size=50, p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# copy originals into augmented folder (so originals are included)
for img_path in glob.glob(os.path.join(train_images, "*.jpg")):
    base = os.path.basename(img_path)
    lbl_path = os.path.join(train_labels, base.replace(".jpg", ".txt"))
    if os.path.exists(lbl_path):
        cv2.imwrite(os.path.join(OUT_IMAGES, base), cv2.imread(img_path))
        with open(lbl_path) as fin:
            open(os.path.join(OUT_LABELS, base.replace(".jpg", ".txt")), "w").write(fin.read())

# generate a small number of augmentations per image (CPU-friendly)
AUG_PER_IMAGE = 2
for img_path in glob.glob(os.path.join(train_images, "*.jpg")):
    base = os.path.basename(img_path)
    img = cv2.imread(img_path)
    lbl_path = os.path.join(train_labels, base.replace(".jpg", ".txt"))
    if not os.path.exists(lbl_path):
        continue
    bboxes = []
    classes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0]); x=float(parts[1]); y=float(parts[2]); w=float(parts[3]); h=float(parts[4])
                bboxes.append([x,y,w,h]); classes.append(cls)
    for i in range(AUG_PER_IMAGE):
        try:
            aug = transform(image=img, bboxes=bboxes, class_labels=classes)
            out_img = aug['image']; out_bboxes = aug['bboxes']; out_classes = aug['class_labels']
            new_base = base.replace(".jpg", f"_aug{i}.jpg")
            cv2.imwrite(os.path.join(OUT_IMAGES, new_base), out_img)
            with open(os.path.join(OUT_LABELS, new_base.replace(".jpg", ".txt")), "w") as fout:
                for cls, bb in zip(out_classes, out_bboxes):
                    fout.write(f"{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
        except Exception:
            # skip images that fail augmentation
            pass

print("Finished augment, folder:", OUT_IMAGES)
