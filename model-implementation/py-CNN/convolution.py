from numpy import matrix
from numpy import multiply
"""
This module is dedicated to implement an common matrix convolution.
"""


class Convolution:
    def embbed(self, image, mask=(3, 3)):
        """
		Create a embbed version of a given image. A embbed version
		is the original image surrounded by a 0's margin with vertical
		tickeness equals to floor(maskHeight/2) and horizontal tickness
		equals to floor(maskWidth/2).

		mask = (height, width)
		"""

        horizTickness = mask[1] // 2
        vertTickness = mask[0] // 2

        embbeded = matrix([[0.0] * (horizTickness * 2 + image.shape[1])
                           for i in range(vertTickness * 2 + image.shape[0])])

        embbeded[horizTickness:(
            embbeded.shape[0] - horizTickness), vertTickness:(
                embbeded.shape[1] - vertTickness)] = image

        return embbeded

    def conv(self, image, mask, embbed=True):
        """
		Returns a filtered version (image (X) mask) of the given image,
		using matrix convolution.
		"""
        refImage = image
        if embbed:
            refImage = self.embbed(image, mask.shape)
        """
		From now and below, is assumed that the image is embbeded.
		"""

        maskHeight, maskWidth = mask.shape
        imageHeight, imageWidth = refImage.shape
        """
		Creates a empty image with the same dimensions as the given 
		image, but without the embbeded strips tickness.
		"""
        filteredImage = matrix(
            [[0.0] * (imageWidth - 2 * (maskWidth // 2))
             for i in range(imageHeight - 2 * (maskHeight // 2))])
        """
		Flipping the kernel, in order to make convolution correctly.
		"""
        flippedKernel = mask[::-1, ::-1]

        # Apply matrix convolution
        for i in range(filteredImage.shape[0]):
            for j in range(filteredImage.shape[1]):
                """
				Important note:
				'multiply' is a element-wise multiplication, 
				not a matrix multiplication.
				"""
                filteredImage[i, j] = multiply(
                    refImage[i:(i + maskHeight), j:(j + maskWidth)],
                    flippedKernel).sum()

        return filteredImage


"""
Program Driver for intern testing.
"""
if __name__ == '__main__':
    c = Convolution()
    i = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    f = matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    print(c.conv(i, f))

    i = matrix([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8]])
    f = matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(c.conv(i, f))
