import tensorflow as tf
import os


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if os.path.exists(log_dir):
            raise Exception("Log directory already exists.")
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)
            self.writer.flush()
        # summary = tf.Summary(
        #     value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step)
                self.writer.flush()
        # summary = tf.Summary(value=[tf.summary.scalar(
        #     tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        # self.writer.add_summary(summary, step)
