import numpy as np
import cv2

def per_channel_color_histogram(image, color_space, quantization_interval):
    bins = 256 // quantization_interval
    if color_space == "RGB":
        blue = np.zeros(bins, int)
        green = np.zeros(bins, int)
        red = np.zeros(bins, int)

        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                blue[image[row, col, 0] // quantization_interval] += 1
                green[image[row, col, 1] // quantization_interval] += 1
                red[image[row, col, 2] // quantization_interval] += 1

        blue = blue / np.linalg.norm(blue, 1)
        green = green / np.linalg.norm(green, 1)
        red = red / np.linalg.norm(red, 1)
    
        return red, green, blue

    elif color_space == "HSV":
        image = image / 255
        Cmax = 0
        Cmin = 0
        dC = 0
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                R = image[row, col, 2]
                G = image[row, col, 1]
                B = image[row, col, 0]
                Cmax = max(R, G, B)
                Cmin = min(R, G, B)
                dC = Cmax - Cmin
                if dC == 0:
                    H = 0
                elif Cmax == R:
                    H = 1. / 6 * ((R - G) / dC % 6)
                elif Cmax == G:
                    H = 1. / 6 * ((G - B) / dC + 2)
                elif Cmax == B:
                    H = 1. / 6 * ((B - R) / dC + 4) 
                if Cmax == 0:
                    S = 0
                elif Cmax > 0:
                    S = dC / Cmax
                V = Cmax
                image[row, col, 2] = H
                image[row, col, 1] = S
                image[row, col, 0] = V
        image = image * 255
        hue = np.zeros(bins, int)
        saturation = np.zeros(bins, int)
        value = np.zeros(bins, int)

        for row in range(image.shape[0]):
            for col in range(image.shape[1]): 
                value[int(image[row, col, 0] // quantization_interval)] += 1
                saturation[int(image[row, col, 1] // quantization_interval)] += 1
                hue[int(image[row, col, 2] // quantization_interval)] += 1

        value = value / np.linalg.norm(value, 1)
        saturation = saturation / np.linalg.norm(saturation, 1)
        hue = hue / np.linalg.norm(hue, 1)

        return hue, saturation, value

def threeD_color_histogram(image, color_space, quantization_interval):
    bins = 256 // quantization_interval
    if color_space == "RGB":
        histogram = np.zeros((bins, bins, bins), int)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                blue = image[row, col, 0] // quantization_interval
                green = image[row, col, 1] // quantization_interval
                red = image[row, col, 2] // quantization_interval
                histogram[blue, green, red] += 1

        histogram = histogram / np.linalg.norm(np.ravel(histogram), 1)
        return histogram

    elif color_space == "HSV":
        image = image / 255
        Cmax = 0
        Cmin = 0
        dC = 0
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                R = image[row, col, 2]
                G = image[row, col, 1]
                B = image[row, col, 0]
                Cmax = max(R, G, B)
                Cmin = min(R, G, B)
                dC = Cmax - Cmin
                if dC == 0:
                    H = 0
                elif Cmax == R:
                    H = 1. / 6 * ((R - G) / dC % 6)
                elif Cmax == G:
                    H = 1. / 6 * ((G - B) / dC + 2)
                elif Cmax == B:
                    H = 1. / 6 * ((B - R) / dC + 4) 
                if Cmax == 0:
                    S = 0
                elif Cmax > 0:
                    S = dC / Cmax
                V = Cmax
                image[row, col, 2] = H
                image[row, col, 1] = S
                image[row, col, 0] = V
        image = image * 255
        histogram = np.zeros((bins, bins, bins), int)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                value = int(image[row, col, 0] // quantization_interval)
                saturation = int(image[row, col, 1] // quantization_interval)
                hue = int(image[row, col, 2] // quantization_interval)
                histogram[value, saturation, hue] += 1

        histogram = histogram / np.linalg.norm(np.ravel(histogram), 1)
        return histogram
    
def similarity_check_per_channel(histogram11, histogram12, histogram13, histogram21, histogram22, histogram23):
    similarity1 = 0
    similarity2 = 0
    similarity3 = 0
    for i in range(len(histogram11)):
        similarity1 += min(histogram11[i], histogram21[i])
        similarity2 += min(histogram12[i], histogram22[i])
        similarity3 += min(histogram13[i], histogram23[i])
    similarity = (similarity1 + similarity2 + similarity3) / 3
    return similarity

def similarity_check_3D(histogram1, histogram2):
    similarity = 0
    histogram1 = histogram1.reshape(histogram1.shape[0]*histogram1.shape[1]*histogram1.shape[2])
    histogram2 = histogram2.reshape(histogram2.shape[0]*histogram2.shape[1]*histogram2.shape[2])
    for i in range(len(histogram1)):
        similarity += min(histogram1[i], histogram2[i])
    return similarity

def configuration(histogram_type, color_space, grid_size, quantization_interval):
    image_names = open("InstanceNames.txt").read().splitlines()
    if grid_size == 1: # non grid based feature extraction
        if histogram_type == "per_channel":
            support_1 = []
            support_2 = []
            support_3 = []
            accuracy1 = 0
            accuracy2 = 0
            accuracy3 = 0
            for i in image_names:
                support_image = cv2.imread("support_96/" + i)
                histogram1, histogram2, histogram3 = per_channel_color_histogram(support_image, color_space, quantization_interval)
                support_1.append(histogram1)
                support_2.append(histogram2)
                support_3.append(histogram3)
            for i in image_names:
                image1 = cv2.imread("query_1/" + i)
                image2 = cv2.imread("query_2/" + i)
                image3 = cv2.imread("query_3/" + i)
                similarity_score1 = []
                similarity_score2 = []
                similarity_score3 = []
                histogram11, histogram12, histogram13 = per_channel_color_histogram(image1, color_space, quantization_interval)
                histogram21, histogram22, histogram23 = per_channel_color_histogram(image2, color_space, quantization_interval)
                histogram31, histogram32, histogram33 = per_channel_color_histogram(image3, color_space, quantization_interval)
                for j in range(len(image_names)):
                    similarity1 = similarity_check_per_channel(histogram11, histogram12, histogram13, support_1[j], support_2[j], support_3[j])
                    similarity2 = similarity_check_per_channel(histogram21, histogram22, histogram23, support_1[j], support_2[j], support_3[j])
                    similarity3 = similarity_check_per_channel(histogram31, histogram32, histogram33, support_1[j], support_2[j], support_3[j])
                    similarity_score1.append(similarity1)
                    similarity_score2.append(similarity2)
                    similarity_score3.append(similarity3)
                index1 = np.argmax(similarity_score1)
                index2 = np.argmax(similarity_score2)
                index3 = np.argmax(similarity_score3)
                if i == image_names[index1]:
                    accuracy1 += 1
                if i == image_names[index2]:
                    accuracy2 += 1
                if i == image_names[index3]:
                    accuracy3 += 1

            accuracy1 /= 200
            accuracy2 /= 200
            accuracy3 /= 200

            print("Accuracy for query_1 is: " + str(accuracy1) + " (for interval: " + str(quantization_interval) + ")")
            print("Accuracy for query_2 is: " + str(accuracy2) + " (for interval: " + str(quantization_interval) + ")")
            print("Accuracy for query_3 is: " + str(accuracy3) + " (for interval: " + str(quantization_interval) + ")")
            return accuracy1, accuracy2, accuracy3

        elif histogram_type == "3D":
            support_histogram = []
            accuracy1 = 0
            accuracy2 = 0
            accuracy3 = 0
            for i in image_names:
                support_image = cv2.imread("support_96/" + i)
                histogram = threeD_color_histogram(support_image, color_space, quantization_interval)
                support_histogram.append(histogram)
            for i in image_names:
                image1 = cv2.imread("query_1/" + i)
                image2 = cv2.imread("query_2/" + i)
                image3 = cv2.imread("query_3/" + i)
                similarity_score1 = []
                similarity_score2 = []
                similarity_score3 = []
                histogram1 = threeD_color_histogram(image1, color_space, quantization_interval)
                histogram2 = threeD_color_histogram(image2, color_space, quantization_interval)
                histogram3 = threeD_color_histogram(image3, color_space, quantization_interval)
                for j in range(len(image_names)):
                    similarity1 = similarity_check_3D(histogram1, support_histogram[j])
                    similarity2 = similarity_check_3D(histogram2, support_histogram[j])
                    similarity3 = similarity_check_3D(histogram3, support_histogram[j])
                    similarity_score1.append(similarity1)
                    similarity_score2.append(similarity2)
                    similarity_score3.append(similarity3)
                index1 = np.argmax(similarity_score1)
                index2 = np.argmax(similarity_score2)
                index3 = np.argmax(similarity_score3)
                if i == image_names[index1]:
                    accuracy1 += 1
                if i == image_names[index2]:
                    accuracy2 += 1
                if i == image_names[index3]:
                    accuracy3 += 1

            accuracy1 /= 200
            accuracy2 /= 200
            accuracy3 /= 200

            print("Accuracy for query_1 is: " + str(accuracy1) + " (for interval: " + str(quantization_interval) + ")")
            print("Accuracy for query_2 is: " + str(accuracy2) + " (for interval: " + str(quantization_interval) + ")")
            print("Accuracy for query_3 is: " + str(accuracy3) + " (for interval: " + str(quantization_interval) + ")")
            return accuracy1, accuracy2, accuracy3

    else: # grid based feature extraction
        grid_length = 96 // grid_size
        if histogram_type == "per_channel":
            support_1 = []
            support_2 = []
            support_3 = []
            accuracy1 = 0
            accuracy2 = 0
            accuracy3 = 0
            for i in range(len(image_names)):
                support_1.append([])
                support_2.append([])
                support_3.append([])
                support_image = cv2.imread("support_96/" + image_names[i])
                for j in range(grid_size):
                    for k in range(grid_size):
                        grid = support_image[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                        grid_histogram1, grid_histogram2, grid_histogram3 = per_channel_color_histogram(grid, color_space, quantization_interval)
                        support_1[i].append(grid_histogram1)
                        support_2[i].append(grid_histogram2)
                        support_3[i].append(grid_histogram3)
            for i in image_names:
                image1 = cv2.imread("query_1/" + i)
                image2 = cv2.imread("query_2/" + i)
                image3 = cv2.imread("query_3/" + i)
                similarity_score1 = []
                similarity_score2 = []
                similarity_score3 = []
                for l in range(len(image_names)):
                    similarity1 = 0
                    similarity2 = 0
                    similarity3 = 0
                    for j in range(grid_size):
                        for k in range(grid_size):
                            grid1 = image1[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                            grid2 = image2[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                            grid3 = image3[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                            histogram11, histogram12, histogram13 = per_channel_color_histogram(grid1, color_space, quantization_interval)
                            histogram21, histogram22, histogram23 = per_channel_color_histogram(grid2, color_space, quantization_interval)
                            histogram31, histogram32, histogram33 = per_channel_color_histogram(grid3, color_space, quantization_interval)
                            sim1 = similarity_check_per_channel(histogram11, histogram12, histogram13, support_1[l][grid_size * j + k], support_2[l][grid_size * j + k], support_3[l][grid_size * j + k])
                            sim2 = similarity_check_per_channel(histogram21, histogram22, histogram23, support_1[l][grid_size * j + k], support_2[l][grid_size * j + k], support_3[l][grid_size * j + k])
                            sim3 = similarity_check_per_channel(histogram31, histogram32, histogram33, support_1[l][grid_size * j + k], support_2[l][grid_size * j + k], support_3[l][grid_size * j + k])
                            similarity1 += sim1
                            similarity2 += sim2
                            similarity3 += sim3
                    similarity1 = similarity1 / (grid_size * grid_size)
                    similarity2 = similarity2 / (grid_size * grid_size)
                    similarity3 = similarity3 / (grid_size * grid_size)
                    similarity_score1.append(similarity1)
                    similarity_score2.append(similarity2)
                    similarity_score3.append(similarity3)
                index1 = np.argmax(similarity_score1)
                index2 = np.argmax(similarity_score2)
                index3 = np.argmax(similarity_score3)
                if i == image_names[index1]:
                    accuracy1 += 1
                if i == image_names[index2]:
                    accuracy2 += 1
                if i == image_names[index3]:
                    accuracy3 += 1

            accuracy1 /= 200
            accuracy2 /= 200
            accuracy3 /= 200

            print("Accuracy for query_1 is: " + str(accuracy1) + " (for interval: " + str(quantization_interval) + " and for spatial grid: " + str(grid_size) + "x" + str(grid_size) + ")")
            print("Accuracy for query_2 is: " + str(accuracy2) + " (for interval: " + str(quantization_interval) + " and for spatial grid: " + str(grid_size) + "x" + str(grid_size) + ")")
            print("Accuracy for query_3 is: " + str(accuracy3) + " (for interval: " + str(quantization_interval) + " and for spatial grid: " + str(grid_size) + "x" + str(grid_size) + ")")
            return accuracy1, accuracy2, accuracy3
            
        elif histogram_type == "3D":
            support_histogram = []
            accuracy1 = 0
            accuracy2 = 0
            accuracy3 = 0
            for i in range(len(image_names)):
                support_histogram.append([])
                support_image = cv2.imread("support_96/" + image_names[i])
                for j in range(grid_size):
                    for k in range(grid_size):
                        grid = support_image[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                        grid_histogram = threeD_color_histogram(grid, color_space, quantization_interval)
                        support_histogram[i].append(grid_histogram)
            for i in image_names:
                image1 = cv2.imread("query_1/" + i)
                image2 = cv2.imread("query_2/" + i)
                image3 = cv2.imread("query_3/" + i)
                similarity_score1 = []
                similarity_score2 = []
                similarity_score3 = []
                for l in range(len(image_names)):
                    similarity1 = 0
                    similarity2 = 0
                    similarity3 = 0
                    for j in range(grid_size):
                        for k in range(grid_size):
                            grid1 = image1[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                            grid2 = image2[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                            grid3 = image3[j * grid_length : (j + 1) * grid_length, k * grid_length : (k + 1) * grid_length]
                            histogram1 = threeD_color_histogram(grid1, color_space, quantization_interval)
                            histogram2 = threeD_color_histogram(grid2, color_space, quantization_interval)
                            histogram3 = threeD_color_histogram(grid3, color_space, quantization_interval)
                            sim1 = similarity_check_3D(histogram1, support_histogram[l][grid_size * j + k])
                            sim2 = similarity_check_3D(histogram2, support_histogram[l][grid_size * j + k])
                            sim3 = similarity_check_3D(histogram3, support_histogram[l][grid_size * j + k])
                            similarity1 += sim1
                            similarity2 += sim2
                            similarity3 += sim3
                    similarity1 = similarity1 / (grid_size * grid_size)
                    similarity2 = similarity2 / (grid_size * grid_size)
                    similarity3 = similarity3 / (grid_size * grid_size)
                    similarity_score1.append(similarity1)
                    similarity_score2.append(similarity2)
                    similarity_score3.append(similarity3)
                index1 = np.argmax(similarity_score1)
                index2 = np.argmax(similarity_score2)
                index3 = np.argmax(similarity_score3)
                if i == image_names[index1]:
                    accuracy1 += 1
                if i == image_names[index2]:
                    accuracy2 += 1
                if i == image_names[index3]:
                    accuracy3 += 1

            accuracy1 /= 200
            accuracy2 /= 200
            accuracy3 /= 200

            print("Accuracy for query_1 is: " + str(accuracy1) + " (for interval: " + str(quantization_interval) + " and for spatial grid: " + str(grid_size) + "x" + str(grid_size) + ")")
            print("Accuracy for query_2 is: " + str(accuracy2) + " (for interval: " + str(quantization_interval) + " and for spatial grid: " + str(grid_size) + "x" + str(grid_size) + ")")
            print("Accuracy for query_3 is: " + str(accuracy3) + " (for interval: " + str(quantization_interval) + " and for spatial grid: " + str(grid_size) + "x" + str(grid_size) + ")")
            return accuracy1, accuracy2, accuracy3
        

### inputs for configuration function: 
## histogram_type: "3D" or "per_channel"
## color_space: "RGB" or "HSV"
## grid_size: 1 for non grid (1x1), 2 for 2x2, 4 for 4x4, 6 for 6x6, 8 for 8x8
## quantization_interval: 8, 16, 32, 64, 128 for experiments

# configuration("3D", "RGB", 1, 16) # Experiment for 3D color histogram with RGB color space
# configuration("3D", "RGB", 1, 32) # Experiment for 3D color histogram with RGB color space
# configuration("3D", "RGB", 1, 64) # Experiment for 3D color histogram with RGB color space
# configuration("3D", "RGB", 1, 128) # Experiment for 3D color histogram with RGB color space

# configuration("3D", "HSV", 1, 16) # Experiment for 3D color histogram with HSV color space 
# configuration("3D", "HSV", 1, 32) # Experiment for 3D color histogram with HSV color space 
# configuration("3D", "HSV", 1, 64) # Experiment for 3D color histogram with HSV color space 
# configuration("3D", "HSV", 1, 128) # Experiment for 3D color histogram with HSV color space 

# configuration("per_channel", "RGB", 1, 8) # Experiment for per-channel color histogram with RGB color space
# configuration("per_channel", "RGB", 1, 16) # Experiment for per-channel color histogram with RGB color space
# configuration("per_channel", "RGB", 1, 32) # Experiment for per-channel color histogram with RGB color space
# configuration("per_channel", "RGB", 1, 64) # Experiment for per-channel color histogram with RGB color space
# configuration("per_channel", "RGB", 1, 128) # Experiment for per-channel color histogram with RGB color space

# configuration("per_channel", "HSV", 1, 8) # Experiment for per-channel color histogram with HSV color space
# configuration("per_channel", "HSV", 1, 16) # Experiment for per-channel color histogram with HSV color space
# configuration("per_channel", "HSV", 1, 32) # Experiment for per-channel color histogram with HSV color space
# configuration("per_channel", "HSV", 1, 64) # Experiment for per-channel color histogram with HSV color space
# configuration("per_channel", "HSV", 1, 128) # Experiment for per-channel color histogram with HSV color space

# configuration("3D", "HSV", 2, 64) # Experiment for 3D color histogram with HSV color space and 2x2 spatial grid
# configuration("3D", "HSV", 4, 64) # Experiment for 3D color histogram with HSV color space and 4x4 spatial grid
# configuration("3D", "HSV", 6, 64) # Experiment for 3D color histogram with HSV color space and 6x6 spatial grid
# configuration("3D", "HSV", 8, 64) # Experiment for 3D color histogram with HSV color space and 8x8 spatial grid

# configuration("per_channel", "HSV", 2, 8) # Experiment for per-channel color histogram with HSV color space and 2x2 spatial grid
# configuration("per_channel", "HSV", 4, 8) # Experiment for per-channel color histogram with HSV color space and 4x4 spatial grid
# configuration("per_channel", "HSV", 6, 8) # Experiment for per-channel color histogram with HSV color space and 6x6 spatial grid
# configuration("per_channel", "HSV", 8, 8) # Experiment for per-channel color histogram with HSV color space and 8x8 spatial grid

# configuration("3D", "RGB", 2, 16) # Experiment for 3D color histogram with RGB color space and 2x2 spatial grid
# configuration("3D", "RGB", 4, 16) # Experiment for 3D color histogram with RGB color space and 4x4 spatial grid
# configuration("3D", "RGB", 6, 16) # Experiment for 3D color histogram with RGB color space and 6x6 spatial grid
# configuration("3D", "RGB", 8, 16) # Experiment for 3D color histogram with RGB color space and 8x8 spatial grid

# configuration("per_channel", "RGB", 2, 16) # Experiment for per-channel color histogram with RGB color space and 2x2 spatial grid
# configuration("per_channel", "RGB", 4, 16) # Experiment for per-channel color histogram with RGB color space and 4x4 spatial grid
# configuration("per_channel", "RGB", 6, 16) # Experiment for per-channel color histogram with RGB color space and 6x6 spatial grid
# configuration("per_channel", "RGB", 8, 16) # Experiment for per-channel color histogram with RGB color space and 8x8 spatial grid
