import numpy as np
import abc
from typing import Dict, Optional, Callable, List
import math


def exponential_learning_rate_decay(base_lr, decay_rate, decay_steps, staircase):
    def call(round_i):
        if staircase:
            return base_lr * (decay_rate ** (round_i // decay_steps))
        else:
            return base_lr * (decay_rate ** (round_i / decay_steps))
    return call


def inverse_linear_decay_learning_rate(base_lr, decay_rate, decay_steps, staircase):
    def call(round_i):
        if staircase:
            return base_lr / (1.0 + decay_rate * (round_i // decay_steps))
        else:
            return base_lr / (1.0 + decay_rate * (round_i / decay_steps))
    return call


def inverse_sqrt_decay_learning_rate(base_lr, decay_rate, decay_steps, staircase):
    def call(round_i):
        if staircase:
            return base_lr / math.sqrt(1.0 + decay_rate * (round_i // decay_steps))
        else:
            return base_lr / math.sqrt(1.0 + decay_rate * (round_i / decay_steps))
    return call


def warmup_learning_rate(base_lr: float, warmup_steps: int, fn: Callable[[int], float]) -> Callable[[int], float]:
    if warmup_steps is None or warmup_steps <= 0:
        def call(round_i) -> float:
            return fn(round_i)
        return call
    else:
        def warmup_and_decay_fn(round_num) -> float:
            if round_num < warmup_steps:
                return base_lr * (round_num + 1) / warmup_steps  # TODO 这个的效果似乎并不好
            else:
                return fn(round_num - warmup_steps)
        return warmup_and_decay_fn


class AdaptiveOptimizer(abc.ABC):
    """
    基于论文 Adaptive Federated Optimization实现的服务端的优化器
    """

    def __init__(self,
                 lr=1.0,
                 lr_decay_policy: str='constant',
                 decay_rate=None,
                 decay_steps=None,
                 staircase=False,
                 warmup_steps=None):
        assert lr_decay_policy == 'constant', 'Only support constant lr scheduler'
        if lr_decay_policy == 'constant':
            scheduler = warmup_learning_rate(lr, warmup_steps=warmup_steps, fn=lambda _: lr)
        elif lr_decay_policy == 'exp_decay':
            scheduler = warmup_learning_rate(lr, warmup_steps, exponential_learning_rate_decay(lr, decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase))
        elif lr_decay_policy == 'inv_lin':
            scheduler = warmup_learning_rate(lr, warmup_steps, inverse_linear_decay_learning_rate(lr, decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase))
        elif lr_decay_policy == 'inv_sqrt':
            scheduler = warmup_learning_rate(lr, warmup_steps, inverse_sqrt_decay_learning_rate(lr, decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase))
        else:
            raise ValueError(
                'Unrecognized schedule type {!s}'.format(lr_decay_policy))
        self.scheduler = scheduler
        self.base_lr = lr

    @abc.abstractmethod
    def step_pseudo_grads(self,
                          round_i: int,
                          global_model: Dict[str, np.ndarray],
                          pseudo_grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass
    
    
class SGD(AdaptiveOptimizer):

    def __init__(self,
                 lr=1.0,
                 lr_decay_policy: str = 'constant',
                 momentum=0,
                 decay_rate=None,
                 decay_steps=None,
                 staircase=False,
                 warmup_steps=None):
        
        super(SGD, self).__init__(lr=lr,
                                  lr_decay_policy=lr_decay_policy,
                                  decay_rate=decay_rate,
                                  decay_steps=decay_steps,
                                  staircase=staircase,
                                  warmup_steps=warmup_steps)
        self.momentum = momentum
        self.buffer = dict()

    def step_pseudo_grads(self,
                          round_i: int,
                          global_model: Dict[str, np.ndarray],
                          pseudo_grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = dict()
        for p_name, p in global_model.items():
            grad = -pseudo_grads[p_name]
            if self.momentum != 0:
                if p_name not in self.buffer:
                    # 复制当前的参数
                    # v_{t} = grad
                    grad = self.buffer[p_name] = np.copy(grad)
                else:
                    # v_{t+1} = v_t * momentum - grad
                    # TODO 这里的实现方式来自于 pytorch 和 tf(https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/SGD) 略有不同
                    self.buffer[p_name] *= self.momentum
                    self.buffer[p_name] += grad
                    grad = self.buffer[p_name]
            # 更新梯度.
            # TODO 对于服务端你的SGD, delta 的格式为 client - init_model, 学习率为 1.0, 模型更新方式是 +(SGD 可以这么实现),
            #  如果调换 delta 的位置, 模型更新就是标准方式, 这也是 FEDADAM, YOGI 等的方式;
            #  但问题是, momentum 计算需要与正的梯度
            # 这里使用 + 法
            result[p_name] = p + (-self.scheduler(round_i)) * grad
        return result
