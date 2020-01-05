#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
python -m tf2onnx.convert : tool to convert a frozen tensorflow graph to onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys

import tensorflow as tf

from tf2onnx.tfonnx import process_tf_graph, tf_optimize
from tf2onnx import constants, loader, logging, utils, optimizer


# pylint: disable=unused-argument

_HELP_TEXT = """
Usage Examples:

python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx
python -m tf2onnx.convert --input frozen_graph.pb  --inputs X:0 --outputs output:0 --output model.onnx
python -m tf2onnx.convert --checkpoint checkpoint.meta  --inputs X:0 --outputs output:0 --output model.onnx

For help and additional information see:
    https://github.com/onnx/tensorflow-onnx

If you run into issues, open an issue here:
    https://github.com/onnx/tensorflow-onnx/issues
"""


def convert_onnx(input_path, inputs_op, outputs_op):

    graphdef = input_path

    if inputs_op:
        inputs_op, shape_override = utils.split_nodename_and_shape(inputs_op)
    if outputs_op:
        outputs_op = outputs_op.split(",")

    logging.basicConfig(level=logging.get_verbosity_level(True))

    utils.set_debug_mode(True)

    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)

    graph_def, inputs_op, outputs_op = loader.from_graphdef(graphdef, inputs_op, outputs_op)
    model_path = graphdef

    graph_def = tf_optimize(inputs_op, outputs_op, graph_def,  True)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(tf_graph,
                             continue_on_error=False,
                             target=",".join(constants.DEFAULT_TARGET),
                             opset=10,
                             custom_op_handlers=None,
                             extra_opset=None,
                             shape_override=None,
                             input_names=inputs_op,
                             output_names=outputs_op,
                             inputs_as_nchw=None)

    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model("converted from {}".format(model_path))

    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model_path)
    # if args.output:
    output_path = input_path.replace(".pb", ".onnx")
    utils.save_protobuf(output_path, model_proto)
    logger.info("ONNX model is saved at %s", output_path)
    # else:
    #     logger.info("To export ONNX model to file, please run with `--output` option")


if __name__ == "__main__":
    convert_onnx(
        input_path="graph.pb",
        inputs_op="input:0",
        outputs_op="dense_decoded:0"
    )
