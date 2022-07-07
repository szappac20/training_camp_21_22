import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_indices, name="baseline"):
        """
        This first task is to predict consumptions one month into the future,
        given the current value of all features.
        The current values include the current consumptions.

        The baseline model just returns the current values of the consumptions
        as the prediction, predicting "No change".

        This is a reasonable baseline whenever the process
        has a slow change rate. slowly.

        Of course, this baseline will work less well
        if you make a prediction further in the future.

        Args:
            label_indices (list): list of integer indices
                                  where the labels are encoded
            name (str): string for the model name
        """
        super().__init__(name=name)
        self.label_indices = label_indices

    def call(self, inputs):
        """
        Method to evaluate the model over a given dataset of inputs

        Args:
            inputs (np.array): numpy array encoding a time series process

        Returns the predictions corresponding to the passed inputs
        """
        if self.label_indices is None:
            return inputs
        result = tf.gather(inputs, indices=self.label_indices, axis=2)
        return result

    def compile_and_fit(self, window):
        """
        This class is required for compliance with Keras Model.
        The mean absolute error metric will be assigned in evaluation mode.
        
        The parameter window is not used.
        There is no fit, since the model has no trainable parameters.

        Args:
            window : a window generator

        """
        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])
