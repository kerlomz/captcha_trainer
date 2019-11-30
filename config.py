#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import platform
import re
import yaml
# import utils
from category import *
from constants import *
from exception import exception, ConfigException

# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# If you have a GPU, you shouldn't care about AVX support.
# Just disables the warning, doesn't enable AVX/FMA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PLATFORM = platform.system()
PATH_SPLIT = "\\" if PLATFORM == "Windows" else "/"
MODEL_CONFIG_NAME = "model.yaml"
IGNORE_FILES = ['.DS_Store']

NETWORK_MAP = {
    'CNNX': CNNNetwork.CNNX,
    'CNN5': CNNNetwork.CNN5,
    'CNNm6': CNNNetwork.CNNm6,
    'CNNm4': CNNNetwork.CNNm4,
    'ResNet': CNNNetwork.ResNet,
    'DenseNet': CNNNetwork.DenseNet,
    'LSTM': RecurrentNetwork.LSTM,
    'BiLSTM': RecurrentNetwork.BiLSTM,
    'GRU': RecurrentNetwork.GRU,
    'BiGRU': RecurrentNetwork.BiGRU,
    'LSTMcuDNN': RecurrentNetwork.LSTMcuDNN,
    'BiLSTMcuDNN': RecurrentNetwork.BiLSTMcuDNN,
    'GRUcuDNN': RecurrentNetwork.GRUcuDNN,
    'NoRecurrent': RecurrentNetwork.NoRecurrent
}

OPTIMIZER_MAP = {
    'AdaBound': Optimizer.AdaBound,
    'Adam': Optimizer.Adam,
    'Momentum': Optimizer.Momentum,
    'SGD': Optimizer.SGD,
    'AdaGrad': Optimizer.AdaGrad,
    'RMSProp': Optimizer.RMSProp
}

MODEL_SCENE_MAP = {
    'Classification': ModelScene.Classification
}

LOSS_FUNC_MAP = {
    'CTC': LossFunction.CTC,
    'CrossEntropy': LossFunction.CrossEntropy
}

RESIZE_MAP = {
    LossFunction.CTC: lambda x, y: [None, y],
    LossFunction.CrossEntropy: lambda x, y: [x, y]
}

LABEL_FROM_MAP = {
    'XML': LabelFrom.XML,
    'LMDB': LabelFrom.LMDB,
    'FileName': LabelFrom.FileName,
}

EXCEPT_FORMAT_MAP = {
    ModelField.Image: 'png',
    ModelField.Text: 'csv'
}

MODEL_FIELD_MAP = {
    'Image': ModelField.Image,
    'Text': ModelField.Text
}

PLATFORM = platform.system()

MODEL_CONFIG_DEMO_NAME = 'model_demo.yaml'


class ModelConfig:

    def __init__(self, project_name, project_path=None):

        self.project_path = project_path if project_path else "./projects/{}".format(project_name)
        self.model_root_path = os.path.join(self.project_path, 'model')
        self.model_conf_path = os.path.join(self.project_path, MODEL_CONFIG_NAME)
        self.output_path = os.path.join(self.project_path, 'out')
        self.dataset_root_path = os.path.join(self.project_path, 'dataset')
        if not os.path.exists(self.model_conf_path):
            self.new()

        """MODEL"""
        self.model_root: dict = self.conf['Model']
        self.model_name: str = self.model_root.get('ModelName')
        self.model_tag = '{model_name}.model'.format(model_name=self.model_name)

        self.model_field_param: str = self.model_root.get('ModelField')
        self.model_field: ModelField = ModelConfig.param_convert(
            source=self.model_field_param,
            param_map=MODEL_FIELD_MAP,
            text="Current model field ({model_field}) is not supported".format(model_field=self.model_field_param),
            code=ConfigException.MODEL_FIELD_NOT_SUPPORTED
        )

        self.model_scene_param: str = self.model_root.get('ModelScene')

        self.model_scene: ModelScene = ModelConfig.param_convert(
            source=self.model_scene_param,
            param_map=MODEL_SCENE_MAP,
            text="Current model scene ({model_scene}) is not supported".format(model_scene=self.model_scene_param),
            code=ConfigException.MODEL_SCENE_NOT_SUPPORTED
        )

        """SYSTEM"""
        self.checkpoint_tag = 'checkpoint'
        self.system_root: dict = self.conf['System']
        self.memory_usage: float = self.system_root.get('MemoryUsage')
        self.save_model: str = os.path.join(self.model_root_path, self.model_tag)
        self.save_checkpoint: str = os.path.join(self.model_root_path, self.checkpoint_tag)

        """FIELD PARAM - IMAGE"""
        self.field_root: dict = self.conf['FieldParam']
        self.category_param = self.field_root.get('Category')
        self.category_value = category_extract(self.category_param)
        self.category: list = SPACE_TOKEN + self.category_value
        self.category_num: int = len(self.category)
        # self.num_classes: int = self.category_num + 2
        self.image_channel: int = self.field_root.get('ImageChannel')
        self.image_width: int = self.field_root.get('ImageWidth')
        self.image_height: int = self.field_root.get('ImageHeight')
        self.resize: list = self.field_root.get('Resize')
        self.max_label_num: int = self.field_root.get('MaxLabelNum')

        """NEURAL NETWORK"""
        self.neu_network_root: dict = self.conf['NeuralNet']
        self.neu_cnn_param = self.neu_network_root.get('CNNNetwork')
        self.neu_cnn: CNNNetwork = ModelConfig.param_convert(
            source=self.neu_cnn_param,
            param_map=NETWORK_MAP,
            text="This neural network ({param}) is not supported at this time.".format(param=self.neu_cnn_param),
            code=ConfigException.NETWORK_NOT_SUPPORTED
        )
        self.neu_recurrent = self.neu_network_root.get('RecurrentNetwork')
        self.num_hidden = self.neu_network_root.get('HiddenNum')
        self.neu_optimizer = self.neu_network_root.get('Optimizer')
        self.neu_optimizer_param: str = self.neu_optimizer if self.neu_optimizer else 'AdaBound'
        self.neu_optimizer: Optimizer = ModelConfig.param_convert(
            source=self.neu_optimizer_param,
            param_map=OPTIMIZER_MAP,
            text="This optimizer ({param}) is not supported at this time.".format(param=self.neu_optimizer_param),
            code=ConfigException.NETWORK_NOT_SUPPORTED
        )
        self.output_layer: dict = self.neu_network_root.get('OutputLayer')
        self.loss_func_param: str = self.output_layer.get('LossFunction')
        self.loss_func = self.param_convert(
            source=self.loss_func_param,
            param_map=LOSS_FUNC_MAP,
            text="This type of loss function ({loss}) is not supported at this time.".format(loss=self.loss_func_param),
            code=ConfigException.LOSS_FUNC_NOT_SUPPORTED,
        )
        self.decoder: str = self.output_layer.get('Decoder')

        """LABEL"""
        self.label_root: dict = self.conf.get('Label')
        self.label_from_param: str = self.label_root.get('LabelFrom')
        self.label_from: LabelFrom = LABEL_FROM_MAP[self.label_from_param]
        self.extract_regex = self.label_root.get('ExtractRegex')
        self.extract_regex = self.extract_regex if self.extract_regex else ".*?(?=_)"
        self.label_split = self.label_root.get('Split')

        """PATH"""
        self.trains_root: dict = self.conf['Trains']
        self.trains_path = self.trains_root.get('TrainsPath')
        self.validation_path = self.trains_root.get('ValidationPath')
        self.dataset_path = self.trains_root.get('DatasetPath')
        self.validation_set_num = self.trains_root.get('ValidationSetNum')
        self.validation_set_num = self.validation_set_num if self.validation_set_num else 500
        self.has_validation_set = self.validation_path and (
            os.path.exists(self.validation_path) if isinstance(self.validation_path, str) else True
        )
        """TRAINS"""
        self.trains_save_steps = self.trains_root.get('SavedSteps')
        self.trains_validation_steps = self.trains_root.get('ValidationSteps')
        self.trains_end_acc = self.trains_root.get('EndAcc')
        self.trains_end_cost = self.trains_root.get('EndCost')
        self.trains_end_cost = self.trains_end_cost if self.trains_end_cost else 1
        self.trains_end_epochs = self.trains_root.get('EndEpochs')
        self.trains_end_epochs = self.trains_end_epochs if self.trains_end_epochs else 2
        self.trains_learning_rate = self.trains_root.get('LearningRate')
        self.batch_size = self.trains_root.get('BatchSize')
        self.batch_size = self.batch_size if self.batch_size else 64
        self.validation_batch_size = self.trains_root.get('ValidationBatchSize')
        self.validation_batch_size = self.validation_batch_size if self.validation_batch_size else 300

        """DATA AUGMENTATION"""
        self.data_augmentation_root: dict = self.conf['DataAugmentation']
        self.binaryzation = self.data_augmentation_root.get('Binaryzation')
        self.median_blur = self.data_augmentation_root.get('MedianBlur')
        self.gaussian_blur = self.data_augmentation_root.get('GaussianBlur')
        self.equalize_hist = self.data_augmentation_root.get('EqualizeHist')
        self.laplace = self.data_augmentation_root.get('Laplace')
        self.rotate = self.data_augmentation_root.get('Rotate')
        self.warp_perspective = self.data_augmentation_root.get('WarpPerspective')
        self.sp_noise = self.data_augmentation_root.get('PepperNoise')

        """COMPILE_MODEL"""
        self.compile_model_path = os.path.join(self.output_path, '{}{}graph'.format(self.model_name, PATH_SPLIT))
        self.check_field()

    @staticmethod
    def param_convert(source, param_map: dict, text, code, default=None):
        if source is None:
            return default
        if source not in param_map.keys():
            exception(text, code)
        return param_map[source]

    def check_field(self):

        if not os.path.exists(self.model_conf_path):
            exception(
                'Configuration File "{}" No Found. '
                'If it is used for the first time, please copy one from {} as {}'.format(
                    MODEL_CONFIG_NAME,
                    MODEL_CONFIG_DEMO_NAME,
                    MODEL_CONFIG_NAME
                ), ConfigException.MODEL_CONFIG_PATH_NOT_EXIST
            )
        if not os.path.exists(self.model_root_path):
            os.makedirs(self.model_root_path)

        # Check category (for classification)
        if self.model_scene == ModelScene.Classification and self.category == ConfigException.CATEGORY_NOT_EXIST:
            exception(
                "The category set type does not exist, there is no character set named {}".format(self.category_param),
                ConfigException.CATEGORY_NOT_EXIST
            )

        model_file = ModelConfig.checkpoint(self.model_name, self.model_root_path)
        checkpoint = 'model_checkpoint_path: {}\nall_model_checkpoint_paths: {}'.format(model_file, model_file)
        with open(self.save_checkpoint, 'w') as f:
            f.write(checkpoint)

    @staticmethod
    def checkpoint(_name, _path):
        file_list = os.listdir(_path)
        checkpoint_group = [
            '"{}"'.format(i.split(".meta")[0]) for i in file_list if
            _name + ".model" in i and i.endswith('.meta')
        ]
        if not checkpoint_group:
            return None
        checkpoint_step = [int(re.search('(?<=model-).*?(?=")', i).group()) for i in checkpoint_group]
        return checkpoint_group[checkpoint_step.index(max(checkpoint_step))]

    @property
    def conf(self) -> dict:
        with open(self.model_conf_path, 'r', encoding="utf-8") as sys_fp:
            sys_stream = sys_fp.read()
            return yaml.load(sys_stream, Loader=yaml.SafeLoader)

    def new(self):
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)

        if not os.path.exists(self.model_root_path):
            os.makedirs(self.model_root_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(self.dataset_root_path):
            os.makedirs(self.dataset_root_path)

        with open("model.template", encoding="utf8") as f:
            base_config = "".join(f.readlines())
            model = base_config.format(
                MemoryUsage=0.7,
                CNNNetwork=CNNNetwork.CNN5.value,
                RecurrentNetwork=RecurrentNetwork.NoRecurrent.value,
                HiddenNum=64,
                Optimizer=Optimizer.AdaBound.value,
                LossFunction='CTC',
                Decoder='CTC',
                ModelName='MyModelName',
                ModelField=ModelField.Image.value,
                ModelScene=ModelScene.Classification.value,
                Category=SimpleCharset.ALPHANUMERIC_LOWER.value,
                Resize=[150, 50],
                ImageChannel=1,
                ImageWidth=150,
                ImageHeight=50,
                MaxLabelNum=-1,
                LabelFrom=LabelFrom.FileName.value,
                ExtractRegex='.*?(?=_)',
                Split='null',
                TrainsPath='',
                ValidationPath='',
                DatasetPath='trains_path',
                ValidationSetNum=300,
                SavedSteps=100,
                ValidationSteps=500,
                EndAcc=0.95,
                EndCost=0.1,
                EndEpochs=2,
                BatchSize=64,
                ValidationBatchSize=300,
                LearningRate=0.001,
                Binaryzation=-1,
                MedianBlur=-1,
                GaussianBlur=-1,
                EqualizeHist=True,
                Laplace=True,
                WarpPerspective=True,
                Rotate=-1,
                PepperNoise=0.1,
            )
        if os.path.exists(self.model_conf_path):
            exception("Already exists {}, unable to create a new profile", -6651)
        else:
            with open(self.model_conf_path, "w", encoding="utf8") as f:
                f.write(model)

    def println(self):
        print('Loading Configuration...')
        print('---------------------------------------------------------------------------------')
        print("PROJECT_PATH", self.project_path)
        print('MODEL_PATH:', self.save_model)
        print('COMPILE_MODEL_PATH:', self.compile_model_path)
        print('CATEGORY_NUM:', self.category_num)
        print('IMAGE_WIDTH: {}, IMAGE_HEIGHT: {}'.format(
            self.image_width, self.image_height)
        )
        print('NEURAL NETWORK: {}'.format(self.neu_network_root))

        print('---------------------------------------------------------------------------------')


if __name__ == '__main__':
    name = "demo"
    c = ModelConfig(project_name=name)
    c.println()

    # trains_path = [
    #     r"H:\Task\验证码\验证码图片内容 参赛数据\train",
    #     # r"C:\Users\kerlomz\Documents\Tencent Files\27009583\FileRecv\图片3\图片"
    # ]
    # trains_path = "".join(["\n    - " + i for i in trains_path])
    # with open("model.template", encoding="utf8") as f:
    #     base_config = "".join(f.readlines())
    #     model = base_config.format(
    #         MemoryUsage=0.7,
    #         CNNNetwork=CNNNetwork.CNN5.value,
    #         RecurrentNetwork=RecurrentNetwork.NoRecurrent.value,
    #         HiddenNum=64,
    #         Optimizer=Optimizer.AdaBound.value,
    #         LossFunction='CTC',
    #         Decoder='CTC',
    #         ModelName='MyModelName',
    #         ModelField=ModelField.Image.value,
    #         ModelScene=ModelScene.Classification.value,
    #         Category=SimpleCharset.ALPHANUMERIC_LOWER.value,
    #         Resize=[150, 50],
    #         ImageChannel=1,
    #         ImageWidth=150,
    #         ImageHeight=50,
    #         LabelFrom=LabelFrom.FileName.value,
    #         ExtractRegex='.*?(?=_)',
    #         Split=None,
    #         TrainsPath='',
    #         ValidationPath='',
    #         DatasetPath=trains_path,
    #         ValidationSetNum=300,
    #         SavedSteps=100,
    #         ValidationSteps=500,
    #         EndAcc=0.95,
    #         EndCost=0.1,
    #         EndEpochs=2,
    #         BatchSize=64,
    #         VerificationBatchSize=300,
    #         LearningRate=0.001
    #     )
    #     print(model)
