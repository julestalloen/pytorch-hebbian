def extract_image_patches(x, kernel_size, stride=(1, 1), dilation=1, padding=0):
    # TODO: implement dilation and padding
    # TODO: does the order in which the patches are returned matter?
    b, c, h, w = x.shape

    # Extract patches
    patches = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

    return patches.view(-1, kernel_size[0], kernel_size[1])
