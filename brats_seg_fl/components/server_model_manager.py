import logging
import os
import re

import numpy as np
import tensorflow as tf

from fed_learn.fed_utils import (FED_DELTA_W, feed_vars,
                                 make_feedable)
from fed_learn.protos.federated_pb2 import ModelData
from fed_learn.server.model_aggregator import ModelAggregator


class ServerModelManager(object):
    """
    Global model manager lives on the server side.
    """

    def __init__(self,
                 start_round=0,
                 num_rounds=-1,
                 keep_round_model=True,
                 exclude_vars=None,
                 model_log_dir=None,
                 ckpt_preload_path=None,
                 model_aggregator=None):
        """
        Start a tensorflow graph and initialise the trainable variables.

        This function saves the initialised model and sets `self.model`
        variable.
        """
        # TODO: loading checkpoint and start from round N
        self.current_round = start_round
        self.num_rounds = num_rounds
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.aggregate_type = FED_DELTA_W

        self.logger = logging.getLogger('ServerModelManager')
        self.model_log_dir = model_log_dir
        self.ckpt_preload_path = ckpt_preload_path
        self.session_restored = False
        self.keep_round_model = keep_round_model

        self.model = None
        self.session = self.get_session()
        # Default max_to_keep is 5
        self._model_saver = tf.train.Saver(save_relative_paths=True, max_to_keep=num_rounds if keep_round_model else 5)
        if keep_round_model:
            self.logger.info('Save every global model!')
        self._ckpt_save_path = os.path.join(self.model_log_dir, 'FL_global_model.ckpt')
        self._best_ckpt_save_path = os.path.join(self.model_log_dir, 'FL_global_model.best.ckpt')

        self.make_init_proto()

        # self.model_aggregator = ModelAggregator(
        #     exclude_vars=exclude_vars
        # )
        self.model_aggregator = model_aggregator

    def get_session(self):
        """
        Create a TF session for the server
        Loading existing checkpoint if required.
        """
        session = tf.Session(graph=tf.get_default_graph())

        # restore session if necessary
        ckpt_path = self.ckpt_preload_path
        if ckpt_path is not None:
            if os.path.isdir(ckpt_path):
                checkpoint = tf.train.get_checkpoint_state(ckpt_path)
                if checkpoint and tf.train.checkpoint_exists(
                        checkpoint.model_checkpoint_path):
                    ckpt_path = checkpoint.model_checkpoint_path
                    self.logger.debug(
                        'Found checkpoint at {}'.format(ckpt_path))
                else:
                    raise ValueError(
                        'cannot find checkpoint at {}'.format(ckpt_path))

            # let tf report error if it can't find the model.ckpt files
            ckpt_loader = tf.train.Saver(
                var_list=None)  # restores all variables
            ckpt_loader.restore(session, ckpt_path)
            self.logger.info('RESTORING ALL VARS from {}'.format(ckpt_path))
            self.session_restored = True
        else:
            self.logger.info('CLEAN START (global_variables_initializer)')

        return session

    @property
    def should_stop(self):
        """
        num_rounds < 0 means non-stopping
        """
        return (self.num_rounds > 0) and (self.current_round >=
                                          self.num_rounds)

    def make_init_proto(self):
        """
        Convert initialised model into protobuf message.
        This function sets self.model to a ModelData protobuf message.
        """
        model_data = ModelData()  # create an empty message
        # session = tf.Session(graph=self.tf_graph)
        # with self.session as sess:

        self.initialize_uninitialized(self.session)
        var_list = tf.global_variables()
        assert var_list, 'Unable to initialize model parameters.'
        var_dict = {var.name: var for var in var_list}
        if self.exclude_vars:
            for var_name in tuple(var_dict):
                if self.exclude_vars.search(var_name):
                    del var_dict[var_name]
        value_dict = self.session.run(var_dict)
        for v_name in value_dict:
            model_data.params[v_name].CopyFrom(
                tf.make_tensor_proto(value_dict[v_name]))

        self.model = model_data

    def save_checkpoint(self, is_best, round_i=None):
        """
        Save the model as a checkpoint.
        """

        self.logger.info('Saving model checkpoint at: {}'.format(self._ckpt_save_path))
        make_feedable(self.session.graph)
        feed_vars(self.session, self.model.params)

        self._model_saver.save(sess=self.session, save_path=self._ckpt_save_path, global_step=round_i)

        if is_best:
            self._model_saver.save(
                sess=self.session, save_path=self._best_ckpt_save_path, global_step=round_i)

    def run_validation(self):
        """
        Run validation
        """
        # if current_round matches the validation interval
        pass

    # def aggregate(self, accumulator):
    #     """
    #     Aggregate tensorflow variables.
    #     This function is not thread-safe.
    #     """
    #     self.logger.info('aggregating %s updates at round %s',
    #                      len(accumulator), self.current_round)
    #     acc_vars = [set(acc.data.params) for acc in accumulator]
    #     acc_vars = set.union(*acc_vars)
    #     # update vars that are not in exclude pattern
    #     vars_to_aggregate = [
    #         g_var for g_var in acc_vars if not self.exclude_vars.search(g_var)
    #     ] if self.exclude_vars else acc_vars
    #     # # only update the intersection set of acc_vars and self.model vars
    #     # vars_to_aggregate = [
    #     #     g_var for g_var in self.model.params if g_var in acc_vars
    #     # ]
    #     if self.aggregate_type != FED_DELTA_W:
    #         raise NotImplementedError('only delta_w aggregation supported.')
    #
    #     for v_name in vars_to_aggregate:
    #         n_local_iters, np_vars = [], []
    #         for acc in accumulator:
    #             if v_name not in acc.data.params:
    #                 continue  # this acc doesn't have the variable from client
    #             float_n_iter = np.float(acc.n_iter)
    #             n_local_iters.append(float_n_iter)
    #             np_vars.append(
    #                 tf.make_ndarray(acc.data.params[v_name]) * float_n_iter)
    #         if not n_local_iters:
    #             continue  # all acc didn't receive the variable from clients
    #         new_val = np.sum(np_vars, axis=0) / np.sum(n_local_iters)
    #         new_val += tf.make_ndarray(self.model.params[v_name])
    #         self.model.params[v_name].CopyFrom(tf.make_tensor_proto(new_val))
    #
    #     # Always save the latest checkpoint
    #     # if self.should_stop:
    #     self.save_checkpoint()
    #
    #     # increase the round number at the end, to avoid the thread racing condition issue.
    #     self.current_round += 1

    def aggregate(self, accumulator):
        """
        Aggregate tensorflow variables.
        This function is not thread-safe.
        """
        self.logger.info('aggregating %s updates at round %s',
                         len(accumulator), self.current_round)

        is_best = self.model_aggregator.aggregate(accumulator, self.model)

        # Always save the latest checkpoint
        # if self.should_stop:
        if self.keep_round_model:
            self.save_checkpoint(is_best, self.current_round)
        else:
            self.save_checkpoint(is_best)

        # increase the round number at the end, to avoid the thread racing condition issue.
        self.current_round += 1

    def initialize_uninitialized(self, session):
        global_vars = tf.global_variables()
        is_not_initialized = session.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        if len(not_initialized_vars):
            session.run(tf.variables_initializer(not_initialized_vars))

    def close(self):
        """
        TODO final saving before quitting
        """
        self.session.close()

        self.logger.info('closing the model manager')