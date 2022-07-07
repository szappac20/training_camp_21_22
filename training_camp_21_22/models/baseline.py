import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_indices, name="baseline"):
        super().__init__(name=name)
        self.label_indices = label_indices

    def call(self, inputs):
        if self.label_indices is None:
            return inputs
        result = tf.gather(inputs, indices=self.label_indices, axis=2)
        return result

    def compile_and_fit(self, window):
        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])
