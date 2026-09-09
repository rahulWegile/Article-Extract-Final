"""
Trimmed vendored copy of UTRNet's model definition, keeping only the
HRNet + double-BiLSTM + CTC configuration this engine actually loads
(UTRNet-Large's published weights use exactly this configuration --
see saved_models/UTRNet-Large in the original repo). The upstream
model.py additionally wires up seven other feature extractors
(DenseNet, InceptionUNet, RCNN, ResNet, ResUnet, AttnUNet, plain
UNet, UNet++, VGG), three other sequence models (LSTM, GRU, MDLSTM),
and an attention-based prediction head -- none of which
UTRNet-Large's checkpoint exercises, so they are not vendored here.

Original: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed
Documents", ICDAR 2023 (Abdur Rahman, Arjun Ghosh, Chetan Arora).
License: CC BY-NC-4.0 -- noncommercial use only (see repo root
requirements.txt comment for this project's own usage terms).
"""

import numpy as np
import torch
import torch.nn as nn

from pipeline.ocr.utrnet.hrnet import HRNet


class HRNetFeatureExtractor(nn.Module):

    def __init__(self, input_channel=1, output_channel=32):

        super().__init__()

        self.ConvNet = HRNet(input_channel, output_channel)

    def forward(self, input):

        return self.ConvNet(input)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):

        self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(input)

        return self.linear(recurrent)


class DropoutLayer(nn.Module):
    """
    UTRNet's own "temporal dropout": zeroes whole timesteps (not
    individual features) of the sequence, and -- unlike a normal
    dropout layer -- fires in EVAL mode too, where UTRNet-Large's
    forward pass runs five independently-dropped copies through the
    sequence model and averages them (see Model.forward below). This
    is part of the trained architecture's inference behavior, not a
    training-only regularizer, so it must be kept exactly as
    published for the pretrained weights to produce the accuracy
    they were measured at.
    """

    def __init__(self, device):

        super().__init__()

        self.device = device

    def forward(self, input):

        keep = (
            np.random.rand(input.shape[1]) > 0.2
        ).astype(int)

        mask = torch.from_numpy(keep).to(self.device)

        mask = mask.reshape(
            input.shape[1], 1
        ).repeat(
            input.shape[0], 1, input.shape[2]
        ).to(self.device)

        return input * mask


class Model(nn.Module):
    """
    HRNet feature extractor -> AdaptiveAvgPool -> temporal dropout
    ensemble -> double BiLSTM -> CTC linear head. Matches
    UTRNet-Large's published architecture exactly (FeatureExtraction
    ="HRNet", SequenceModeling="DBiLSTM", Prediction="CTC" in the
    original repo's terms) -- `opt` only needs input_channel,
    output_channel, hidden_size, num_class, device.
    """

    def __init__(self, opt):

        super().__init__()

        self.opt = opt

        self.FeatureExtraction = HRNetFeatureExtractor(
            opt.input_channel, opt.output_channel,
        )

        self.FeatureExtraction_output = opt.output_channel

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.dropout1 = DropoutLayer(opt.device)
        self.dropout2 = DropoutLayer(opt.device)
        self.dropout3 = DropoutLayer(opt.device)
        self.dropout4 = DropoutLayer(opt.device)
        self.dropout5 = DropoutLayer(opt.device)

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(
                self.FeatureExtraction_output,
                opt.hidden_size,
                opt.hidden_size,
            ),
            BidirectionalLSTM(
                opt.hidden_size,
                opt.hidden_size,
                opt.hidden_size,
            ),
        )

        self.SequenceModeling_output = opt.hidden_size

        self.Prediction = nn.Linear(
            self.SequenceModeling_output, opt.num_class,
        )

    def forward(self, input):

        visual_feature = self.FeatureExtraction(input)

        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2)
        )

        visual_feature = visual_feature.squeeze(3)

        if self.training:

            dropped = self.dropout1(visual_feature)

            contextual_feature = self.SequenceModeling(dropped)

        else:

            # Five independently-dropped passes, averaged -- see
            # DropoutLayer's docstring for why this runs in eval too.
            branches = [
                self.SequenceModeling(
                    dropout(visual_feature)
                )
                for dropout in (
                    self.dropout1,
                    self.dropout2,
                    self.dropout3,
                    self.dropout4,
                    self.dropout5,
                )
            ]

            contextual_feature = sum(branches) / len(branches)

        return self.Prediction(contextual_feature.contiguous())
