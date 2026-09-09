"""
Trimmed vendored copy of UTRNet's label conversion and image
normalization helpers -- only the CTC converter (this engine's
Prediction head) and the aspect-ratio-preserving pad/normalize
transform used at inference time. The upstream utils.py/dataset.py
additionally define an attention-based converter, an LMDB training
dataset pipeline, and a non-padding resize transform, none of which
this engine (inference-only, CTC) uses.

Original: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
License: CC BY-NC-4.0 -- noncommercial use only.
"""

import numpy as np
import torch


class CTCLabelConverter:
    """ Convert between text-label and text-index. """

    def __init__(self, character):

        dict_character = list(character)

        self.dict = {}

        for i, char in enumerate(dict_character):
            # 0 is reserved for the CTC blank token.
            self.dict[char] = i + 1

        self.character = ["[CTCblank]"] + dict_character

    def decode(self, text_index, length):

        texts = []

        for index, l in enumerate(length):

            t = text_index[index, :]

            char_list = []

            for i in range(l):

                # Drop blanks and collapse repeated characters.
                if t[i] != 0 and not (i > 0 and t[i - 1] == t[i]):
                    char_list.append(self.character[t[i]])

            texts.append("".join(char_list))

        return texts


def to_tensor(image):
    """
    Grayscale PIL image -> CHW float tensor in [-1, 1], the same
    normalization torchvision.transforms.ToTensor() + a (0.5, 0.5)
    sub/div would produce -- implemented directly on numpy so this
    engine does not need torchvision as a dependency.
    """

    array = np.asarray(image, dtype=np.float32) / 255.0

    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    else:
        array = array.transpose(2, 0, 1)

    tensor = torch.from_numpy(array)

    tensor.sub_(0.5).div_(0.5)

    return tensor


class NormalizePAD:
    """
    Resizes (aspect-ratio preserved, height fixed) then right-pads a
    single-channel image tensor to `max_size` = (channels, H, W),
    replicating the last real column into the padding -- matches
    UTRNet's own training-time transform exactly, which the
    published checkpoint's weights expect at inference time too.
    """

    def __init__(self, max_size):

        self.max_size = max_size

    def __call__(self, image):

        tensor = to_tensor(image)

        c, h, w = tensor.size()

        padded = torch.zeros(*self.max_size, dtype=tensor.dtype)

        padded[:, :, :w] = tensor

        if self.max_size[2] != w:

            padded[:, :, w:] = (
                tensor[:, :, w - 1]
                .unsqueeze(2)
                .expand(c, h, self.max_size[2] - w)
            )

        return padded
