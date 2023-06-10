import numpy as np
from PIL import Image

# Entry point
def main():
	img = Image.open('C:/Users/zyh/Desktop/85051.png').convert('L')
	data = np.uint8(np.array(img))
	eqData = localEq(data, 11)
	img = Image.fromarray(eqData)
	img.save('C:/Users/zyh/Desktop/our.png')


# Transformation helper function
def calcTransform(data):
    nPixels = data.size
    hist, _ = np.histogram(data, bins=255 + 1, range=(0, 255))
    assert hist.sum() == nPixels, 'Unexpected input data format!'
    # Store the transformation in s_k
    s_k = np.zeros(255 + 1)
    for idx, _ in enumerate(s_k):
        s_k[idx] = 255 / nPixels * hist[:idx + 1].sum()
    # Round the transformation down to nearest integer
    s_k = np.floor(s_k)
    return s_k

# Local Histogram equalization
def localEq(data, n):
    eqData = np.zeros(data.shape, np.uint8)
    tmp = list()

    # Loop over each pixel in the image
    for (x, y), value in np.ndenumerate(data):
        # Collect the local neighborhood into tmp
        for i in range(0, n):
            s_x = x - int(n / 2) + i
            for j in range(0, n):
                s_y = y - int(n / 2) + j
                if s_x >= 0 and s_x < data.shape[0] and s_y >= 0 and s_y < data.shape[1]:
                    tmp.append(data[s_x][s_y])

        # Calculate the histogram transformation
        s_k = calcTransform(np.asarray(tmp))

        # Lookup the transformation for the given pixel
        eqData[x, y] = s_k[value]

        # Clear tmp for the next iteration
        tmp = []

    return eqData



if __name__ == '__main__':
    main()