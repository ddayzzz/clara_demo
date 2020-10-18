"""Server and client side model handlers"""

import logging
import re

import numpy as np
import tensorflow as tf

from fed_learn.client.fed_privacy import PercentileProtocol, SVTProtocol
from fed_learn.fed_utils import (FED_DELTA_W, feed_vars,
                                 make_feedable)
from fed_learn.protos.federated_pb2 import ModelData, ModelMetaData


class ClientModelManager():
    """
    Client-side model manager lives on client's local machine.
    """

    def __init__(self, task_names=None, exclude_vars=None,
                 privacy=None):
        """
        Start a tensorflow graph and a tensorflow session for the client.

        :param task_names: a list of hashable, each uniquely defines
             a remote server
        """
        self.logger = logging.getLogger('ClientModelManager')

        self.federated_meta = {task_name: list() for task_name in task_names}
        # a regular expression string, matching the
        # variable names. Matched variables will be client-specific
        # variables, not read to or written by the server's global model
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.fitter = None

        # self.logger.info('privacy params: %s', privacy_params)
        # self.privacy_policy = None
        # if privacy_params:
        #     _dp_type = privacy_params.pop('dp_type', None)
        #     if _dp_type == 'partial':
        #         self.privacy_policy = PercentileProtocol(**privacy_params)
        #     elif _dp_type == 'laplacian':
        #         self.privacy_policy = SVTProtocol(**privacy_params)
        self.privacy_policy = privacy
        if self.privacy_policy is None:
            self.logger.info('privacy module disabled.')

    def set_fitter(self, fitter):
        if self.fitter is not None:
            return
        make_feedable(fitter.graph)
        new_sess = tf.Session(graph=fitter.graph, config=fitter.tf_config)
        with new_sess.as_default(), fitter.graph.as_default():
            fitter.initialize()
        self.fitter = fitter

    def num_local_iterations(self):
        """
        Number of local training iterations per federated round

        :return: 1 if fitter not set else value extracted from fitter
        """
        if self.fitter is None:
            return 1
        return self.fitter.train_ctx.total_steps

    def model_meta(self, task_name):
        """
        task meta data, should be consistent with the server's
        """
        task_meta = self.federated_meta.get(task_name, [])
        return ModelMetaData() if not task_meta else task_meta[0]

    def model_vars(self, task_name):
        """
        task variables, should be a subset of the server's
        """
        task_meta = self.federated_meta.get(task_name, [])
        return [] if not task_meta else task_meta[1]

    def read_current_model(self, task_name, type_str=FED_DELTA_W):
        """
        Read variables from the local tensorflow session.

        :return: a ModelData message
        """
        model_vars = self.model_vars(task_name)
        # could grep by var names
        with self.fitter.graph.as_default():
            all_vars = tf.global_variables()
            var_list = [
                var for var in all_vars if var.name in tuple(model_vars)
            ]
            var_values = self.fitter.session.run(var_list)
        var_list = [var.name for var in var_list]

        local_model_dict = dict(zip(var_list, var_values))
        if type_str != FED_DELTA_W:
            raise NotImplementedError('Only delta_w supported.')
        global_model_dict = {
            var_name: tf.make_ndarray(model_vars[var_name])
            for var_name in tuple(model_vars)
        }
        # compute delta model, global model has the primary key set
        model_diff = {}
        for name in global_model_dict:
            if name not in local_model_dict:
                continue
            model_diff[name] = local_model_dict[name] - global_model_dict[name]
        # modifying delta_w
        if self.privacy_policy is not None:
            model_diff = self.privacy_policy.apply(model_diff,
                                                   self.fitter.train_ctx)

        # prepare output of model reading
        model_data = ModelData()
        model_values = []
        for v_name in model_diff:
            model_data.params[v_name].CopyFrom(
                tf.make_tensor_proto(model_diff[v_name]))
            model_values.append(model_diff[v_name].ravel())
        # invariant to number of steps
        model_values = np.concatenate(model_values) / \
            np.float(self.fitter.train_ctx.total_steps)
        add_tensorboard_histogram(
            self.fitter.summary_writer,
            self.fitter.train_ctx.global_round,
            'model_diff',
            model_values,
            bins=1000)
        return model_data

    def assign_current_model(self, remote_models) -> bool:
        """
        Set the local tf session's model according to model_data
        :param remote_models: 模型, 从不同 servers 端接收的(目前仅仅支持一台服务器), 元素类型为 a ModelData message
        :return: True 表示赋值完成
        """

        if all([model is None for model in remote_models]):
            return False

        if all([model.allowed_to_perform_update == 0 for model in remote_models]):
            return False

        changed_vars = []
        for model in remote_models:
            if not model:
                continue
            remote_task_name = model.meta.task.name

            # filtering the variables to be shared
            model_params = model.data.params
            for name in tuple(model_params):
                if self.exclude_vars and self.exclude_vars.search(name):
                    del model_params[name]

            # tracking remote model variable names
            self.federated_meta[remote_task_name].clear()
            self.federated_meta[remote_task_name].append(model.meta)
            self.federated_meta[remote_task_name].append(model_params)

            assign_ops = feed_vars(self.fitter.session, model_params)
            changed_vars.extend(assign_ops)
        if not changed_vars:
            return False  # no vars matched from the remote_models
        num_vars = 0
        for changed_var in changed_vars:
            num_vars += np.prod(changed_var.shape)
        self.logger.info(
            'Setting graph with global federated model data (%s elements)',
            num_vars)
        if remote_models[0] is not None:
            self.logger.info('Round %s: local model updated',
                             remote_models[0].meta.current_round)

        if all([model.allowed_to_perform_update == 2 for model in remote_models]):
            return False

        return True

    def train(self):
        return self.fitter.fit()

    def close(self):
        if self.fitter is not None:
            self.fitter.close()


def add_tensorboard_histogram(tf_writer, step, tag, values, bins=1000):
    """
    add numpy to tensorboard histogram
    """
    values = np.array(values)
    if not np.isfinite(values).all():
        return
    counts, bin_edges = np.histogram(values, bins=bins)

    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    bin_edges = bin_edges[1:]
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for value_c in counts:
        hist.bucket.append(value_c)

    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    tf_writer.add_summary(summary, step)
    tf_writer.flush()
