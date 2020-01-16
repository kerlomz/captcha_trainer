#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import yaml
import json


class ModelConfig:

    def __init__(self, model_conf: str):
        self.model_conf = model_conf
        self.system = None
        self.device = None
        self.device_usage = None
        self.charset = None
        self.split_char = None
        self.gen_charset = None
        self.char_exclude = None
        self.model_name = None
        self.model_type = None
        self.image_height = None
        self.image_width = None
        self.image_channel = None
        self.padding = None
        self.lower_padding = None
        self.resize = None
        self.binaryzation = None
        self.smooth = None
        self.blur = None
        self.replace_transparent = None
        self.model_site = None
        self.version = None
        self.color_engine = None
        self.cf_model = self.read_conf
        self.model_exists = False
        self.assignment()

    def assignment(self):

        system = self.cf_model.get('System')
        self.device = system.get('Device') if system else None
        self.device = self.device if self.device else "cpu:0"
        self.device_usage = system.get('DeviceUsage') if system else None
        self.device_usage = self.device_usage if self.device_usage else 0.01
        self.charset = self.cf_model['Model'].get('CharSet')
        self.char_exclude = self.cf_model['Model'].get('CharExclude')
        self.model_name = self.cf_model['Model'].get('ModelName')
        self.model_type = self.cf_model['Model'].get('ModelType')
        self.model_site = self.cf_model['Model'].get('Sites')
        self.model_site = self.model_site if self.model_site else []
        self.version = self.cf_model['Model'].get('Version')
        self.version = self.version if self.version else 1.0
        self.split_char = self.cf_model['Model'].get('SplitChar')
        self.split_char = '' if not self.split_char else self.split_char

        self.image_height = self.cf_model['Model'].get('ImageHeight')
        self.image_width = self.cf_model['Model'].get('ImageWidth')
        self.image_channel = self.cf_model['Model'].get('ImageChannel')
        self.image_channel = self.image_channel if self.image_channel else 1
        self.binaryzation = self.cf_model['Pretreatment'].get('Binaryzation')
        self.resize = self.cf_model['Pretreatment'].get('Resize')
        self.resize = self.resize if self.resize else [self.image_width, self.image_height]
        self.replace_transparent = self.cf_model['Pretreatment'].get('ReplaceTransparent')

    @property
    def read_conf(self):
        with open(self.model_conf, 'r', encoding="utf-8") as sys_fp:
            sys_stream = sys_fp.read()
            return yaml.load(sys_stream, Loader=yaml.SafeLoader)

    def convert(self):
        with open("../model.template", encoding="utf8") as f:
            lines = f.readlines()
            bc = "".join(lines)
            model = bc.format(
                MemoryUsage=0.7,
                CNNNetwork='CNNX',
                RecurrentNetwork='GRU',
                UnitsNum=64,
                Optimizer='Adam',
                LossFunction='CTC',
                Decoder='CTC',
                ModelName=self.model_name,
                ModelField='Image',
                ModelScene='Classification',
                Category=self.charset,
                Resize=json.dumps(self.resize),
                ImageChannel=self.image_channel,
                ImageWidth=self.image_width,
                ImageHeight=self.image_height,
                MaxLabelNum=4,
                AutoPadding=False,
                OutputSplit="",
                LabelFrom="FileName",
                ExtractRegex=".*?(?=_)",
                LabelSplit='null',
                DatasetTrainsPath="",
                DatasetValidationPath="",
                SourceTrainPath="",
                SourceValidationPath="",
                ValidationSetNum="300",
                SavedSteps="500",
                ValidationSteps="500",
                EndAcc="0.98",
                EndCost="0.05",
                EndEpochs="2",
                BatchSize="64",
                ValidationBatchSize="300",
                LearningRate="0.001",
                DA_Binaryzation="-1",
                DA_MedianBlur="-1",
                DA_GaussianBlur="-1",
                DA_EqualizeHist="False",
                DA_Laplace="False",
                DA_WarpPerspective="False",
                DA_Rotate="-1",
                DA_PepperNoise="-1",
                DA_Brightness="False",
                DA_Saturation="False",
                DA_Hue="False",
                DA_Gamma="False",
                DA_ChannelSwap="False",
                DA_RandomBlank="-1",
                DA_RandomTransition="-1",
                Pre_Binaryzation="-1",
                Pre_ReplaceTransparent="False",
                Pre_HorizontalStitching="False",
                Pre_ConcatFrames="-1",
                Pre_BlendFrames="-1",
            )
        with open(self.model_conf.replace(".yaml", "_2.0.yaml"), "w", encoding="utf8") as f:
            f.write(model)


if __name__ == '__main__':
    ModelConfig(model_conf="model.yaml").convert()
