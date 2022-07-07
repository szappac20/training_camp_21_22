import tensorflow as tf
import cycler
import matplotlib.pyplot as plt
import numpy as np


class WindowGenerator(object):
    def __init__(
            self, input_width, label_width, shift,
            train_df, val_df, test_df, label_columns=None):
        """
        Class to manage the windowing of a time-series dataset
        in terms of sub-sequences of a given length input_width,
        coupled with label_width predictions.

        The predictions are shifted with shift steps
        with respect to the inputs.

        train_df, val_df, test_df must be Pandas dataframes
        with the same columns
        If no evaluation dataset is available the test dataset can be used

        A single-step window with a single prediction for the next time-step
        value may be defined as follows

        >>> w2 = WindowGenerator(
                input_width=1, label_width=1, shift=1,
                train_df=train_df, val_df, test_df,
                label_columns=['f1', 'f2', 'f3'])
        Total window size: 2
        Input indices: [0]
        Label indices: [1]
        Label column name(s): ['f1', 'f2', 'f3']

        Args:
            input_width (int): number of time-steps used for the prediction
            label_width (int): number of time-steps to be predicted
            shift (int): shift between the last time step used for input
                         and the first used for prediction
            train_df (Pandas.DataFrame): train dataset
            val_df (Pandas.DataFrame): evaluation dataset
            test_df (Pandas.DataFrame): test dataset
            label_columns (list): list of string denoting the columns
                                  of the ground truth
                                  (energy consumptions for f1, f2, f3)
        """

        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)}
        self.column_indices = {
            name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(
            self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(
            self.total_window_size)[self.labels_slice]
        self._example = self.example

    def __repr__(self):
        """
        A description of the main aspects of the window
        through the standard output
        """
        return "\n".join([
            f"Total window size: {self.total_window_size}",
            f"Input indices: {self.input_indices}",
            f"Label indices: {self.label_indices}",
            f"Label column name(s): {self.label_columns}"])

    def split_window(self, features):
        """
        Given a list of consecutive inputs,
        this method will convert
        them to a window of inputs and a window of labels.

        The example w2 you define earlier w

        Args:
            features (np.array): numpy 3-dimensional array
                                 [num_batches, total_window_size, num_features]
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([
                labels[:, :, self.column_indices[name]] for name in
                self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_cols=None, max_subplots=3):
        """
        Plot the split window showing the inputs and the labels.
        If a model is passed, the predictions are plotted as well.

        Args:
            model (keras.Model): keras model object
            plot_cols (list): columns to be shown
            max_subplots (int): maximum number of subplots

        """
        c_list = ["blue", "orange", "green"]
        plt.rc("axes", prop_cycle=(cycler.cycler("color", c_list)))

        if plot_cols is None:
            plot_cols = ["f1", "f2", "f3"]

        inputs, labels = self.example

        fig = plt.figure(figsize=(12, 8))
        plot_col_indices = [self.column_indices[col] for col in plot_cols]

        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            # plt.ylabel(f"{plot_cols} [normed]")
            plt.ylabel("kWh")
            plt.plot(
                self.input_indices,
                tf.gather(inputs, indices=plot_col_indices, axis=2)[n, :, :],
                label=plot_cols, marker=".", zorder=-10, alpha=0.5)

            for f in range(len(plot_cols)):
                plt.scatter(
                    x=self.label_indices, y=labels[n, :, f], edgecolors="k",
                    c=c_list[f], s=64, label=f"{plot_cols[f]} measured",
                    alpha=0.5)
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(
                        x=self.label_indices,
                        y=predictions[n, :, f], marker="X", edgecolors="k",
                        c=c_list[f], s=64, label=f"{plot_cols[f]} prediction",
                        alpha=0.5)

            if n == 0:
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.xlabel("Time [h]")
        plt.suptitle(f"{model.name}")
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)

    def make_dataset(self, data, sequence_stride=12, batch_size=32):
        """

        Args:
            data (np.array):
            sequence_stride (int):
            batch_size (int): size of the batch

        Returns:
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            sampling_rate=1,
            shuffle=False,
            batch_size=batch_size,
            start_index=0,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """
        Get and cache an example batch of `inputs, labels` for plotting.
        """
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
