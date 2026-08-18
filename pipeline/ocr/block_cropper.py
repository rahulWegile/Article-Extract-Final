import cv2


class BlockCropper:

    def crop(self, image, block):

        x1 = max(0, int(block.x1))
        y1 = max(0, int(block.y1))
        x2 = min(image.shape[1], int(block.x2))
        y2 = min(image.shape[0], int(block.y2))

        return image[y1:y2, x1:x2]