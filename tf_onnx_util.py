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
from tf2onnx import constants, logging, utils, optimizer
from tf_graph_util import convert_variables_to_constants
# from tensorflow.python.framework.graph_util import convert_variables_to_constants

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

logger = tf.compat.v1.logging
# logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    """Freezes the state of a session into a pruned computation graph."""
    output_names = [i.split(':')[:-1][0] for i in output_names]
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def(add_shapes=True)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(sess, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def remove_redundant_inputs(frozen_graph, input_names):
    """Remove redundant inputs not in frozen graph."""
    frozen_inputs = []
    # get inputs in frozen graph
    for n in frozen_graph.node:
        for inp in input_names:
            if utils.node_name(inp) == n.name:
                frozen_inputs.append(inp)
    deleted_inputs = list(set(input_names) - set(frozen_inputs))
    if deleted_inputs:
        logger.warning("inputs [%s] is not in frozen graph, delete them", ",".join(deleted_inputs))
    return frozen_inputs


def from_graphdef(sess, graph_def, model_path, input_names, output_names):
    """Load tensorflow graph from graphdef."""
    # make sure we start with clean default graph
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        frozen_graph = freeze_session(sess, output_names=output_names)
    input_names = remove_redundant_inputs(frozen_graph, input_names)
    # clean up
    return frozen_graph, input_names, output_names


def convert_onnx(sess, graph_def, input_path, inputs_op, outputs_op):

    graphdef = input_path

    if inputs_op:
        inputs_op, shape_override = utils.split_nodename_and_shape(inputs_op)
    if outputs_op:
        outputs_op = outputs_op.split(",")

    # logging.basicConfig(level=logging.get_verbosity_level(True))

    utils.set_debug_mode(True)

    graph_def, inputs_op, outputs_op = from_graphdef(sess, graph_def, graphdef, inputs_op, outputs_op)
    model_path = graphdef

    graph_def = tf_optimize(inputs_op, outputs_op, graph_def, True)

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


if __name__ == "__main__":
    pass
