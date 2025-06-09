import cv2, os

input_dir  = "dataset/images/train"
output_dir = "dataset/images/train_upright"
os.makedirs(output_dir, exist_ok=True)

for fn in os.listdir(input_dir):
    if not fn.lower().endswith(".jpg"): continue
    img = cv2.imread(os.path.join(input_dir, fn))
    # rotate 90° clockwise
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_dir, fn), img)