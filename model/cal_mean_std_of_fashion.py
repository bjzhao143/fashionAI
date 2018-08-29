from PIL import Image
import numpy as np

TRAIN_IMG_DIR = r'E:\competition\tianchi\fashionAI\train\Images'

def get_files(dir):
    import os
    if os.path.isfile(dir):
        return [dir]
    result = []
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        result += get_files(sub_path)
    return result

r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0

files = get_files(TRAIN_IMG_DIR)
count = len(files)

for i, image_file in enumerate(files):
    print('Process: %d/%d' % (i, count))
    img = Image.open(image_file)
    img = np.asarray(img)
    img = img.astype('float32') / 255.
    total += img.shape[0] * img.shape[1]

    r += img[:, :, 0].sum()
    g += img[:, :, 1].sum()
    b += img[:, :, 2].sum()

    r_2 += (img[:, :, 0] ** 2).sum()
    g_2 += (img[:, :, 1] ** 2).sum()
    b_2 += (img[:, :, 2] ** 2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

################################
##  val = (1/n)*∑(x^2) - M^2  ##
################################
r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2


print('Mean : %s' % ([r_mean, g_mean, b_mean]))
print('Var : %s' % ([r_var, g_var, b_var]))

# Mean : [0.63911890191781129, 0.59919561822653, 0.58661944147125888]
# Var : [0.08388869692697809, 0.087893603904720508, 0.089051650207497945]
