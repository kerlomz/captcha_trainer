#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import json
import platform
import re
import yaml
from category import *
from constants import *
from exception import exception, ConfigException

# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# If you have a GPU, you shouldn't care about AVX support.
# Just disables the warning, doesn't enable AVX/FMA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PLATFORM = platform.system()
# PATH_SPLIT = "\\" if PLATFORM == "Windows" else "/"
PATH_SPLIT = "/"
MODEL_CONFIG_NAME = "model.yaml"
IGNORE_FILES = ['.DS_Store']

NETWORK_MAP = {
    'CNNX': CNNNetwork.CNNX,
    'CNN5': CNNNetwork.CNN5,
    'ResNetTiny': CNNNetwork.ResNetTiny,
    'ResNet50': CNNNetwork.ResNet50,
    'DenseNet': CNNNetwork.DenseNet,
    'MobileNetV2': CNNNetwork.MobileNetV2,
    'LSTM': RecurrentNetwork.LSTM,
    'BiLSTM': RecurrentNetwork.BiLSTM,
    'GRU': RecurrentNetwork.GRU,
    'BiGRU': RecurrentNetwork.BiGRU,
    'LSTMcuDNN': RecurrentNetwork.LSTMcuDNN,
    'BiLSTMcuDNN': RecurrentNetwork.BiLSTMcuDNN,
    'GRUcuDNN': RecurrentNetwork.GRUcuDNN,
    'NoRecurrent': RecurrentNetwork.NoRecurrent
}

BUILT_IN_CATEGORY_MAP = {
    'NUMERIC': SimpleCharset.NUMERIC,
    'ALPHANUMERIC': SimpleCharset.ALPHANUMERIC,
    'ALPHANUMERIC_LOWER': SimpleCharset.ALPHANUMERIC_LOWER,
    'ALPHANUMERIC_UPPER': SimpleCharset.ALPHANUMERIC_UPPER,
    'ALPHABET_LOWER': SimpleCharset.ALPHABET_LOWER,
    'ALPHABET_UPPER': SimpleCharset.ALPHABET_UPPER,
    'ALPHABET': SimpleCharset.ALPHABET,
    'ARITHMETIC': SimpleCharset.ARITHMETIC,
    'FLOAT': SimpleCharset.FLOAT,
    'CHS_3500': SimpleCharset.CHS_3500,
    'ALPHANUMERIC_CHS_3500_LOWER': SimpleCharset.ALPHANUMERIC_CHS_3500_LOWER,
}

OPTIMIZER_MAP = {
    'RAdam': Optimizer.RAdam,
    'Adam': Optimizer.Adam,
    'AdaBound': Optimizer.AdaBound,
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

COMPILE_MODEL_MAP = {
    ModelType.PB: ".pb",
    ModelType.ONNX: ".onnx",
    ModelType.TFLITE: ".tflite"
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

OUTPUT_SHAPE1_MAP = {
    CNNNetwork.CNN5: [16, 64],
    CNNNetwork.CNNX: [8, 64],
    CNNNetwork.ResNetTiny: [16, 1024],
    CNNNetwork.ResNet50: [16, 2048],
    CNNNetwork.DenseNet: [32, 2048],
    CNNNetwork.MobileNetV2: [32, 1200]
}


class DataAugmentationEntity:
    binaryzation: object = -1
    median_blur: int = -1
    gaussian_blur: int = -1
    equalize_hist: bool = False
    laplace: bool = False
    warp_perspective: bool = False
    rotate: int = -1
    sp_noise: float = -1.0
    brightness: bool = False
    saturation: bool = False
    hue: bool = False
    gamma: bool = False
    channel_swap: bool = False
    random_blank: int = -1
    random_transition: int = -1


class PretreatmentEntity:
    binaryzation: object = -1
    concat_frames: object = -1
    blend_frames: object = -1
    replace_transparent: bool = True
    horizontal_stitching: bool = False


class ModelConfig:
    """MODEL"""
    model_root: dict
    model_name: str
    model_tag: str
    model_field_param: str
    model_scene_param: str

    """SYSTEM"""
    system_root: dict
    memory_usage: float
    save_model: str
    save_checkpoint: str

    """FIELD PARAM - IMAGE"""
    field_root: dict
    category_param: list or str
    image_channel: int
    image_width: int
    image_height: int
    resize: list
    max_label_num: int
    auto_padding: bool
    output_split: str

    """NEURAL NETWORK"""
    neu_network_root: dict
    neu_cnn_param: str
    neu_recurrent_param: str
    units_num: int
    neu_optimizer_param: str
    output_layer: dict
    loss_func_param: str
    decoder: str

    """LABEL"""
    label_root: dict
    label_from_param: str
    extract_regex: str
    label_split: str

    """PATH"""
    trains_root: dict
    dataset_path_root: dict
    source_path_root: dict
    trains_path: dict = {DatasetType.TFRecords: [], DatasetType.Directory: []}
    validation_path: dict = {DatasetType.TFRecords: [], DatasetType.Directory: []}
    dataset_map = {
        RunMode.Trains: trains_path,
        RunMode.Validation: validation_path
    }
    validation_set_num: int

    """TRAINS"""
    trains_save_steps: int
    trains_validation_steps: int
    trains_end_acc: float
    trains_end_cost: float
    trains_end_epochs: int
    trains_learning_rate: float
    batch_size: int
    validation_batch_size: int

    """DATA AUGMENTATION"""
    data_augmentation_root: dict
    da_binaryzation: list
    da_median_blur: int
    da_gaussian_blur: int
    da_equalize_hist: bool
    da_laplace: bool
    da_rotate: int
    da_warp_perspective: bool
    da_sp_noise: float
    da_brightness: bool
    da_saturation: bool
    da_hue: bool
    da_gamma: bool
    da_channel_swap: bool
    da_random_blank: int
    da_random_transition: int

    """PRETREATMENT"""
    pretreatment_root: dict
    pre_binaryzation: int
    pre_replace_transparent: bool
    pre_horizontal_stitching: bool
    pre_concat_frames: object
    pre_blend_frames: object

    """COMPILE_MODEL"""
    compile_model_path: str

    def __init__(self, project_name, project_path=None, is_dev=True, **argv):
        self.is_dev = is_dev
        self.project_path = project_path if project_path else "./projects/{}".format(project_name)
        self.output_path = os.path.join(self.project_path, 'out')
        self.compile_conf_path = os.path.join(self.output_path, 'model')
        self.compile_conf_path = os.path.join(self.compile_conf_path, "{}_model.yaml".format(project_name))
        self.model_root_path = os.path.join(self.project_path, 'model')
        self.model_conf_path = os.path.join(self.project_path, MODEL_CONFIG_NAME)
        self.dataset_root_path = os.path.join(self.project_path, 'dataset')
        self.checkpoint_tag = 'checkpoint'

        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)

        if not os.path.exists(self.model_root_path):
            os.makedirs(self.model_root_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(self.dataset_root_path):
            os.makedirs(self.dataset_root_path)

        if len(argv) > 0:
            self.new(**argv)
        else:
            self.read_conf()

    def read_conf(self):
        """MODEL"""
        self.model_root = self.conf['Model']
        self.model_name = self.model_root.get('ModelName')
        self.model_tag = '{model_name}.model'.format(model_name=self.model_name)

        self.model_field_param = self.model_root.get('ModelField')
        self.model_scene_param = self.model_root.get('ModelScene')

        """SYSTEM"""
        self.system_root = self.conf['System']
        self.memory_usage = self.system_root.get('MemoryUsage')
        self.model_version = self.system_root.get("Version")
        self.save_model = os.path.join(self.model_root_path, self.model_tag)
        self.save_checkpoint = os.path.join(self.model_root_path, self.checkpoint_tag)

        """FIELD PARAM - IMAGE"""
        self.field_root = self.conf['FieldParam']
        self.category_param = self.field_root.get('Category')

        self.image_channel = self.field_root.get('ImageChannel')
        self.image_width = self.field_root.get('ImageWidth')
        self.image_height = self.field_root.get('ImageHeight')
        self.resize = self.field_root.get('Resize')
        self.max_label_num = self.field_root.get('MaxLabelNum')
        self.auto_padding = self.field_root.get('AutoPadding')
        self.output_split = self.field_root.get('OutputSplit')

        """NEURAL NETWORK"""
        self.neu_network_root = self.conf['NeuralNet']
        self.neu_cnn_param = self.neu_network_root.get('CNNNetwork')

        self.neu_recurrent_param = self.neu_network_root.get('RecurrentNetwork')
        self.neu_recurrent_param = self.neu_recurrent_param if self.neu_recurrent_param else 'NoRecurrent'

        self.units_num = self.neu_network_root.get('UnitsNum')
        self.neu_optimizer_param = self.neu_network_root.get('Optimizer')
        self.neu_optimizer_param = self.neu_optimizer_param if self.neu_optimizer_param else 'RAdam'

        self.output_layer = self.neu_network_root.get('OutputLayer')
        self.loss_func_param = self.output_layer.get('LossFunction')

        self.decoder = self.output_layer.get('Decoder')

        """LABEL"""
        self.label_root = self.conf.get('Label')
        self.label_from_param = self.label_root.get('LabelFrom')
        self.extract_regex = self.label_root.get('ExtractRegex')
        self.extract_regex = self.extract_regex if self.extract_regex else ".*?(?=_)"
        self.label_split = self.label_root.get('LabelSplit')

        """PATH"""
        self.trains_root = self.conf['Trains']

        self.dataset_path_root = self.trains_root.get('DatasetPath')
        self.trains_path[DatasetType.TFRecords]: list = self.dataset_path_root.get('Training')
        self.validation_path[DatasetType.TFRecords]: list = self.dataset_path_root.get('Validation')

        self.source_path_root = self.trains_root.get('SourcePath')
        self.trains_path[DatasetType.Directory]: list = self.source_path_root.get('Training')
        self.validation_path[DatasetType.Directory]: list = self.source_path_root.get('Validation')

        self.validation_set_num: int = self.trains_root.get('ValidationSetNum')
        # self.validation_set_num = self.validation_set_num if self.validation_set_num else 500

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
        self.data_augmentation_root = self.conf['DataAugmentation']
        self.da_binaryzation = self.data_augmentation_root.get('Binaryzation')
        self.da_median_blur = self.data_augmentation_root.get('MedianBlur')
        self.da_gaussian_blur = self.data_augmentation_root.get('GaussianBlur')
        self.da_equalize_hist = self.data_augmentation_root.get('EqualizeHist')
        self.da_laplace = self.data_augmentation_root.get('Laplace')
        self.da_rotate = self.data_augmentation_root.get('Rotate')
        self.da_warp_perspective = self.data_augmentation_root.get('WarpPerspective')
        self.da_sp_noise = self.data_augmentation_root.get('PepperNoise')
        self.da_brightness = self.data_augmentation_root.get('Brightness')
        self.da_saturation = self.data_augmentation_root.get('Saturation')
        self.da_hue = self.data_augmentation_root.get('Hue')
        self.da_gamma = self.data_augmentation_root.get('Gamma')
        self.da_channel_swap = self.data_augmentation_root.get('ChannelSwap')
        self.da_random_blank = self.data_augmentation_root.get('RandomBlank')
        self.da_random_transition = self.data_augmentation_root.get('RandomTransition')

        """PRETREATMENT"""
        self.pretreatment_root = self.conf['Pretreatment']
        self.pre_binaryzation = self.pretreatment_root.get('Binaryzation')
        self.pre_replace_transparent = self.pretreatment_root.get("ReplaceTransparent")
        self.pre_horizontal_stitching = self.pretreatment_root.get("HorizontalStitching")
        self.pre_concat_frames = self.pretreatment_root.get('ConcatFrames')
        self.pre_blend_frames = self.pretreatment_root.get('BlendFrames')

        """COMPILE_MODEL"""
        self.compile_model_path = os.path.join(self.output_path, 'graph')
        self.compile_model_path = self.compile_model_path.replace("\\", "/")
        self.check_field()

    @property
    def model_field(self) -> ModelField:
        return ModelConfig.param_convert(
            source=self.model_field_param,
            param_map=MODEL_FIELD_MAP,
            text="Current model field ({model_field}) is not supported".format(model_field=self.model_field_param),
            code=ConfigException.MODEL_FIELD_NOT_SUPPORTED
        )

    @property
    def model_scene(self) -> ModelScene:
        return ModelConfig.param_convert(
            source=self.model_scene_param,
            param_map=MODEL_SCENE_MAP,
            text="Current model scene ({model_scene}) is not supported".format(model_scene=self.model_scene_param),
            code=ConfigException.MODEL_SCENE_NOT_SUPPORTED
        )

    @property
    def neu_cnn(self) -> CNNNetwork:
        return ModelConfig.param_convert(
            source=self.neu_cnn_param,
            param_map=NETWORK_MAP,
            text="This cnn layer ({param}) is not supported at this time.".format(param=self.neu_cnn_param),
            code=ConfigException.NETWORK_NOT_SUPPORTED
        )

    @property
    def neu_recurrent(self) -> RecurrentNetwork:
        return ModelConfig.param_convert(
            source=self.neu_recurrent_param,
            param_map=NETWORK_MAP,
            text="Current recurrent layer ({recurrent}) is not supported".format(recurrent=self.neu_recurrent_param),
            code=ConfigException.NETWORK_NOT_SUPPORTED
        )

    @property
    def neu_optimizer(self) -> Optimizer:
        return ModelConfig.param_convert(
            source=self.neu_optimizer_param,
            param_map=OPTIMIZER_MAP,
            text="This optimizer ({param}) is not supported at this time.".format(param=self.neu_optimizer_param),
            code=ConfigException.NETWORK_NOT_SUPPORTED
        )

    @property
    def loss_func(self) -> LossFunction:
        return ModelConfig.param_convert(
            source=self.loss_func_param,
            param_map=LOSS_FUNC_MAP,
            text="This type of loss function ({loss}) is not supported at this time.".format(loss=self.loss_func_param),
            code=ConfigException.LOSS_FUNC_NOT_SUPPORTED,
        )

    @property
    def label_from(self) -> LabelFrom:
        return ModelConfig.param_convert(
            source=self.label_from_param,
            param_map=LABEL_FROM_MAP,
            text="This type of label from ({lf}) is not supported at this time.".format(lf=self.label_from_param),
            code=ConfigException.ERROR_LABEL_FROM,
        )

    @property
    def category(self) -> list:
        category_value = category_extract(self.category_param)
        return SPACE_TOKEN + category_value

    @property
    def category_num(self) -> int:
        return len(self.category)

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
                'If it is used for the first time, please copy one according to model.template as {}'.format(
                    MODEL_CONFIG_NAME,
                    MODEL_CONFIG_NAME
                ), ConfigException.MODEL_CONFIG_PATH_NOT_EXIST
            )
        if not os.path.exists(self.model_root_path):
            os.makedirs(self.model_root_path)

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
        with open(self.model_conf_path if self.is_dev else self.compile_conf_path, 'r', encoding="utf-8") as sys_fp:
            sys_stream = sys_fp.read()
            return yaml.load(sys_stream, Loader=yaml.SafeLoader)

    @staticmethod
    def list_param(params, intent=6):
        if params is None:
            params = []
        if isinstance(params, str):
            params = [params]
        result = "".join(["\n{}- ".format(' ' * intent) + i for i in params])
        return result

    @staticmethod
    def val_filter(val):
        if isinstance(val, str) and len(val) == 1:
            val = "'{}'".format(val)
        elif val is None:
            val = 'null'
        return val

    def update(self, model_conf_path=None, model_name=None):
        with open("model.template", encoding="utf8") as f:
            base_config = "".join(f.readlines())
            model = base_config.format(
                MemoryUsage=self.memory_usage,
                CNNNetwork=self.neu_cnn.value,
                RecurrentNetwork=self.val_filter(self.neu_recurrent_param),
                UnitsNum=self.units_num,
                Optimizer=self.neu_optimizer.value,
                LossFunction=self.loss_func.value,
                Decoder=self.decoder,
                ModelName=model_name if model_name else self.model_name,
                ModelField=self.model_field.value,
                ModelScene=self.model_scene.value,
                Category=self.category_param,
                Resize=json.dumps(self.resize),
                ImageChannel=self.image_channel,
                ImageWidth=self.image_width,
                ImageHeight=self.image_height,
                MaxLabelNum=self.max_label_num,
                AutoPadding=self.auto_padding,
                OutputSplit=self.val_filter(self.output_split),
                LabelFrom=self.label_from.value,
                ExtractRegex=self.val_filter(self.extract_regex),
                LabelSplit=self.val_filter(self.label_split),
                DatasetTrainsPath=self.list_param(self.trains_path[DatasetType.TFRecords], intent=6),
                DatasetValidationPath=self.list_param(self.validation_path[DatasetType.TFRecords], intent=6),
                SourceTrainPath=self.list_param(self.trains_path[DatasetType.Directory], intent=6),
                SourceValidationPath=self.list_param(self.validation_path[DatasetType.Directory], intent=6),
                ValidationSetNum=self.validation_set_num,
                SavedSteps=self.trains_save_steps,
                ValidationSteps=self.trains_validation_steps,
                EndAcc=self.trains_end_acc,
                EndCost=self.trains_end_cost,
                EndEpochs=self.trains_end_epochs,
                BatchSize=self.batch_size,
                ValidationBatchSize=self.validation_batch_size,
                LearningRate=self.trains_learning_rate,
                DA_Binaryzation=self.da_binaryzation,
                DA_MedianBlur=self.da_median_blur,
                DA_GaussianBlur=self.da_gaussian_blur,
                DA_EqualizeHist=self.da_equalize_hist,
                DA_Laplace=self.da_laplace,
                DA_WarpPerspective=self.da_warp_perspective,
                DA_Rotate=self.da_rotate,
                DA_PepperNoise=self.da_sp_noise,
                DA_Brightness=self.da_brightness,
                DA_Saturation=self.da_saturation,
                DA_Hue=self.da_hue,
                DA_Gamma=self.da_gamma,
                DA_ChannelSwap=self.da_channel_swap,
                DA_RandomBlank=self.da_random_blank,
                DA_RandomTransition=self.da_random_transition,
                Pre_Binaryzation=self.pre_binaryzation,
                Pre_ReplaceTransparent=self.pre_replace_transparent,
                Pre_HorizontalStitching=self.pre_horizontal_stitching,
                Pre_ConcatFrames=self.pre_concat_frames,
                Pre_BlendFrames=self.pre_blend_frames,
            )
        with open(model_conf_path if model_conf_path else self.model_conf_path, "w", encoding="utf8") as f:
            f.write(model)

    def output_config(self, target_model_name=None):
        compiled_config_dir_path = os.path.join(self.output_path, "model")
        if not os.path.exists(compiled_config_dir_path):
            os.makedirs(compiled_config_dir_path)
        compiled_config_path = os.path.join(compiled_config_dir_path, "{}_model.yaml".format(self.model_name))
        self.update(model_conf_path=compiled_config_path, model_name=target_model_name)

    def dataset_increasing_name(self, mode: RunMode):
        dataset_group = os.listdir(self.dataset_root_path)
        if len(dataset_group) < 1:
            return "Trains.0.tfrecords" if mode == RunMode.Trains else "Validation.0.tfrecords"
        name_split = [i.split(".") for i in dataset_group if mode.value in i]
        last_index = max([int(i[1]) for i in name_split])
        current_index = last_index + 1
        name_prefix = name_split[0][0]
        name_suffix = name_split[0][2]
        return "{}.{}.{}".format(name_prefix, current_index, name_suffix)

    def new(self, **argv):
        self.memory_usage = argv.get('MemoryUsage')
        self.neu_cnn_param = argv.get('CNNNetwork')
        self.neu_recurrent_param = argv.get('RecurrentNetwork')
        self.units_num = argv.get('UnitsNum')
        self.neu_optimizer_param = argv.get('Optimizer')
        self.loss_func_param = argv.get('LossFunction')
        self.decoder = argv.get('Decoder')
        self.model_name = argv.get('ModelName')
        self.model_field_param = argv.get('ModelField')
        self.model_scene_param = argv.get('ModelScene')

        if isinstance(argv.get('Category'), list):
            self.category_param = json.dumps(argv.get('Category'), ensure_ascii=False)
        else:
            self.category_param = argv.get('Category')

        self.resize = argv.get('Resize')
        self.image_channel = argv.get('ImageChannel')
        self.image_width = argv.get('ImageWidth')
        self.image_height = argv.get('ImageHeight')
        self.max_label_num = argv.get('MaxLabelNum')
        self.auto_padding = argv.get('AutoPadding')
        self.output_split = argv.get('OutputSplit')
        self.label_from_param = argv.get('LabelFrom')
        self.extract_regex = argv.get('ExtractRegex')
        self.label_split = argv.get('LabelSplit')
        self.trains_path[DatasetType.TFRecords] = argv.get('DatasetTrainsPath')
        self.validation_path[DatasetType.TFRecords] = argv.get('DatasetValidationPath')
        self.trains_path[DatasetType.Directory] = argv.get('SourceTrainPath')
        self.validation_path[DatasetType.Directory] = argv.get('SourceValidationPath')
        self.validation_set_num = argv.get('ValidationSetNum')
        self.trains_save_steps = argv.get('SavedSteps')
        self.trains_validation_steps = argv.get('ValidationSteps')
        self.trains_end_acc = argv.get('EndAcc')
        self.trains_end_cost = argv.get('EndCost')
        self.trains_end_epochs = argv.get('EndEpochs')
        self.batch_size = argv.get('BatchSize')
        self.validation_batch_size = argv.get('ValidationBatchSize')
        self.trains_learning_rate = argv.get('LearningRate')
        self.da_binaryzation = argv.get('DA_Binaryzation')
        self.da_median_blur = argv.get('DA_MedianBlur')
        self.da_gaussian_blur = argv.get('DA_GaussianBlur')
        self.da_equalize_hist = argv.get('DA_EqualizeHist')
        self.da_laplace = argv.get('DA_Laplace')
        self.da_warp_perspective = argv.get('DA_WarpPerspective')
        self.da_rotate = argv.get('DA_Rotate')
        self.da_sp_noise = argv.get('DA_PepperNoise')
        self.da_brightness = argv.get('DA_Brightness')
        self.da_saturation = argv.get('DA_Saturation')
        self.da_hue = argv.get('DA_Hue')
        self.da_gamma = argv.get('DA_Gamma')
        self.da_channel_swap = argv.get('DA_ChannelSwap')
        self.da_random_blank = argv.get('DA_RandomBlank')
        self.da_random_transition = argv.get('DA_RandomTransition')
        self.pre_binaryzation = argv.get('Pre_Binaryzation')
        self.pre_replace_transparent = argv.get('Pre_ReplaceTransparent')
        self.pre_horizontal_stitching = argv.get('Pre_HorizontalStitching')
        self.pre_concat_frames = argv.get('Pre_ConcatFrames')
        self.pre_blend_frames = argv.get('Pre_BlendFrames')

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
    c.update()
