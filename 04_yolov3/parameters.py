import numpy as np


class ParameterPool:
    class LOSS:
        STANDARD = {}

    class ANCHOR:
        COCO = np.array([[13, 10], [30, 16], [23, 33],
                         [61, 30], [45, 62], [119, 59],
                         [90, 116], [198, 156], [326, 373]])

