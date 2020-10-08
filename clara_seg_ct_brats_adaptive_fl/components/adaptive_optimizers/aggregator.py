
import logging
import re

import numpy as np
import tensorflow as tf


from fed_learn.fed_utils import (FED_DELTA_W)
from adaptive_optimizers.optimizers import AdaptiveOptimizer, SGD


class AdaptiveAggregator:

    """
    实现 Adaptive Method
    """

    def __init__(self,
                 algo_optimizer=None,
                 exclude_vars=None,
                 server_lr=1.0,
                 algo_kws=None):
        """

        :param algo_optimizer: 服务端的优化器
        :param exclude_vars: 去掉的变量名
        :param algo_kws: 服务端的优化器的参数设定
        """
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.aggregate_type = FED_DELTA_W
        self.server_lr = server_lr

        self.logger = logging.getLogger('AdaptiveAggregator')
        self.logger.info('optimizer select: {0}, initialized with parameters : {1}, server_lr: {2}'.format(algo_optimizer, algo_kws, server_lr))

        if algo_optimizer is not None:
            assert algo_optimizer in ['fedavg', 'fedav', 'fed']
        else:
            algo_optimizer = 'fedavg'
        self.algo_optimizer = algo_optimizer

        if algo_kws is None:
            algo_kws = dict()
        self.algo_kws = algo_kws
        self.server_optimizer = self._choose_server_optimizer()

    def aggregate(self, accumulator, model):
        """
        Aggregate model variables.
        This function is not thread-safe.

        :return Return True to indicate that the current model is the best model so far.
        """

        acc_vars = [set(acc.data.params) for acc in accumulator]
        acc_vars = set.union(*acc_vars)
        # update vars that are not in exclude pattern
        vars_to_aggregate = [
            g_var for g_var in acc_vars if not self.exclude_vars.search(g_var)
        ] if self.exclude_vars else acc_vars
        # # only update the intersection set of acc_vars and self.model vars
        # vars_to_aggregate = [
        #     g_var for g_var in self.model.params if g_var in acc_vars
        # ]
        if self.aggregate_type != FED_DELTA_W:
            raise NotImplementedError('only delta_w aggregation supported.')


        model_params = dict()
        deltas = dict()
        for v_name in vars_to_aggregate:
            n_local_iters, np_vars = [], []
            for acc in accumulator:
                """
                客户端的UID: acc.client.uid
                """
                if v_name not in acc.data.params:
                    continue  # this acc doesn't have the variable from client
                # batch运行的次数
                float_n_iter = np.float(acc.n_iter)
                n_local_iters.append(float_n_iter)
                # weight_delta * n_iter  其中 (weight_delta = client_model - global_model)

                weighted_value = tf.make_ndarray(acc.data.params[v_name]) * float_n_iter

                np_vars.append(weighted_value)
            if not n_local_iters:
                continue  # all acc didn't receive the variable from clients
            # 平均
            new_val = np.sum(np_vars, axis=0) / np.sum(n_local_iters)
            # 这里不是很明白? 我觉得这个过程类似于梯度更新
            ### 普通更新的方法
            # new_val += tf.make_ndarray(model.params[v_name])
            # # 给模型重新赋值
            # model.params[v_name].CopyFrom(tf.make_tensor_proto(new_val))
            ### 创建服务端的优化器
            model_params[v_name] = tf.make_ndarray(model.params[v_name])
            deltas[v_name] = new_val
        new_global_model = self.server_optimizer.step_pseudo_grads(1, model_params, deltas)
        for v_name, p in new_global_model.items():
            model.params[v_name].CopyFrom(tf.make_tensor_proto(p))
        # 永远更新模型
        return False

    def _choose_server_optimizer(self) -> AdaptiveOptimizer:
        if self.algo_optimizer == 'fedavg':
            return SGD(lr=self.server_lr)
        elif self.algo_optimizer == 'fedavgm':
            return SGD(lr=self.server_lr, momentum=self.algo_kws['momentum'])
        # elif self.algo_optimizer == 'fedadam':
        #     return tf.keras.optimizers.Adam(learning_rate=self.server_lr,
        #                                     momentum=self.algo_kws['momentum'],
        #                                     epsilon=self.algo_kws['epsilon'])
        else:
            raise NotImplementedError('Not implementation : {0}'.format(self.algo_optimizer))



