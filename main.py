from utils import *
from solve import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import tensorflow as tf

image = cv2.imread(r'E:\computer-vision\sudoku-solver\images\4.jpg')
img = image.copy()
preprocessed_img = preprocess(img)
contours = findcontours(preprocessed_img)
biggestcontor,area = biggestContour(contours)
reorder_contours = reorder(biggestcontor)
persp_img = fix_perspective(preprocessed_img,reorder_contours)
removed_lines = remove_lines(persp_img)
boards = np.split(removed_lines,9)

model = tf.keras.models.load_model(r'E:\computer-vision\sudoku-solver\digit_model.h5')
grid = []

for i in range(9):
    j = 0
    row = []
    
    while j <= 800:
        img = boards[i][:, j:j+100]
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((3,3), np.uint8)
        # img = cv2.erode(img, kernel, iterations=1)
        img = cv2.resize(img, (28, 28))
        img = np.expand_dims(img, axis=0)
        prob = model.predict(img)

        digit = np.argmax(prob) if prob[0][np.argmax(prob)] >= 0.9 else 0

        row.append(digit)

        # Visualize the digit and its corresponding image
        # plt.imshow(img.reshape(28, 28), cmap='gray')
        # plt.title(f"Predicted Digit: {digit}")
        # plt.show()

        j += 100
    
    grid.append(row)

print(grid)




re_persp_img = cv2.resize(persp_img,(900,900))
org_grid = copy.deepcopy(grid)
solved_puzzle = solve_sudoku(grid)
print(org_grid)
result = overlay(org_grid,solved_puzzle,re_persp_img)
 

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img_rgb)
axs[0].set_title('Original Image')

axs[1].imshow(result_rgb)
axs[1].set_title('Solved!')

plt.show()


