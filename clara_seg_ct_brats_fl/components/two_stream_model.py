import tensorflow as tf
from ai4med.common.graph_component import GraphComponent
from ai4med.common.build_ctx import BuildContext


class Model(GraphComponent):
    """Base class of Models

    Args:
        None
    Returns:
        Prediction results

    """

    def __init__(self):
        GraphComponent.__init__(self)

    def get_loss(self):
        """Get the additional loss function in AHNet model.

        Args:
            None

        Returns:
            Loss function

        """

        return 0

    def get_update_ops(self):
        """Get the update_ops for Batch Normalization.

        The method "tf.control_dependencies" allow the operations used as inputs
        of the context manager are run before the operations defined inside the
        context manager. So we use "update_ops" to implement Batch Normalization.

        Args:
            None

        Returns:
            Update operations

        """

        return tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def get_predictions(self, inputs, is_training, build_ctx: BuildContext):
        """Forward computation process of model for both training and inference.

        Args:
            inputs (tf.Tensor): input data for the AHNet model
            is_training (bool): in training process or not
            build_ctx(BuildContext): reserved argument for future features

        Returns:
            Prediction results

        """

        raise NotImplementedError('Class {} does not implement get_predictions'.format(
            self.__class__.__name__))

    def build(self, build_ctx: BuildContext):
        """Connect model with graph.

        Args:
            build_ctx: specified graph context

        Returns:
            Prediction results

        """

        inputs = build_ctx.must_get(BuildContext.KEY_MODEL_INPUT)
        is_training = build_ctx.must_get(BuildContext.KEY_IS_TRAIN)
        return self.get_predictions(inputs, is_training, build_ctx)