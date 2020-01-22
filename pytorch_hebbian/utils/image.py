import cv2 as cv


def resize_image_dim(shape, max_width=None, max_height=None):
    h, w = shape[:2]

    # If both the width and height are None, then return the original image
    if max_width is None and max_height is None:
        return shape

    # Calculate the ratio of the height and construct the dimensions
    if max_height is not None:
        rh = max_height / float(h)
    else:
        rh = 1

    if max_width is not None:
        rw = max_width / float(w)
    else:
        rw = 1

    if rh < 1 <= rw:
        # The height is the restricting dimension
        dim = (int(w * rh), max_height)
    elif rw < 1 <= rh:
        # The width is the restricting dimension
        dim = (max_width, int(h * rw))
    elif (rh <= 1 and rw <= 1) or (rh >= 1 and rw >= 1):
        # Scale according to the most restricting scaling factor
        if rh < rw:
            dim = (int(w * rh), max_height)
        else:
            dim = (max_width, int(h * rw))
    else:
        dim = (w, h)

    return dim


def resize_image(image, max_width=None, max_height=None, inter=cv.INTER_AREA):
    """Resize an image to the specified maximum dimensions without aspect ratio distortion.

    Basically, this function will attempt to make the image as big as possible, while maintaining aspect ratio,
    without exceeding the specified maximum dimensions.
    """
    dim = resize_image_dim(image.shape[:2], max_width, max_height)
    resized = cv.resize(image, dim, interpolation=inter)

    return resized
