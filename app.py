import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False

# Load your model
MODEL = load_model("best_model.h5.keras")

# Labels for digit classification
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

pygame.init()

# Initialize font
FONT = pygame.font.Font(None, 18)  # Using default font

# Set up display surface
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Handwritten Digit Recognizer")

iswritting = False
Number_xcord = []
Number_ycord = []
image_cnt = 1
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswritting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            Number_xcord.append(xcord)
            Number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswritting = True
        
        if event.type == MOUSEBUTTONUP:
            iswritting = False
            
            # Only proceed if coordinates have been collected
            if Number_xcord and Number_ycord:
                Number_xcord = sorted(Number_xcord)
                Number_ycord = sorted(Number_ycord)

                rect_min_x, rect_max_x = max(Number_xcord[0] - BOUNDRYINC, 0), min(Number_xcord[-1] + BOUNDRYINC, WINDOWSIZEX)
                rect_min_y, rect_max_y = max(Number_ycord[0] - BOUNDRYINC, 0), min(Number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

                # Extract the sub-image from the drawn area
                pixel_array = pygame.PixelArray(DISPLAYSURF)
                img_arr = np.array(pixel_array[rect_min_x:rect_max_x, rect_min_y:rect_max_y]).astype(np.float32)
                img_arr = np.transpose(img_arr)  # Transpose to align correctly
                del pixel_array  # Release the PixelArray

                Number_xcord = []
                Number_ycord = []

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.jpg", img_arr)
                    image_cnt += 1

                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255
                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                    textSurface = FONT.render(label, True, RED, WHITE)
                    textRecOB = textSurface.get_rect()
                    textRecOB.left, textRecOB.bottom = rect_min_x, rect_max_y

                    DISPLAYSURF.blit(textSurface, textRecOB)

            # Reset coordinate lists in case they weren't used
            Number_xcord = []
            Number_ycord = []

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)
                
    pygame.display.update()
