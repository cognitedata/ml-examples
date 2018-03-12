import argparse
import glob
import os

import keras
import pandas as pd
import tensorflow as tf
import blueprint.model as model
from keras.models import load_model
from tensorflow.python.lib.io import file_io

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
MODEL_NAME = 'lstm.hdf5'


class ContinuousEval(keras.callbacks.Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 eval_files,
                 learning_rate,
                 job_dir,
                 pp_dir,
                 steps=1000):
        self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.pp_dir = pp_dir
        self.steps = steps

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0 and epoch % self.eval_frequency == 0:

            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith("gs://"):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                lstm_model = load_model(checkpoints[-1])
                lstm_model = model.compile_model(lstm_model)
                X, y = model.generate_input(self.eval_files, self.pp_dir)
                loss = lstm_model.evaluate(x=X, y=y)
                print('\nEvaluation epoch[{}] loss[{:.2f}] {}'.format(
                    epoch, loss, lstm_model.metrics_names))
                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))


def dispatch(train_files,
             eval_files,
             job_dir,
             pp_dir,
             train_batch_size,
             eval_batch_size,
             learning_rate,
             eval_frequency,
             layers,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs):
    input_shape = (2, (pd.read_csv(tf.gfile.Open(train_files[0])).shape[1] - 2))
    lstm_model = model.model_fn(input_shape, layers)

    try:
        os.makedirs(job_dir)
    except:
        pass

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=checkpoint_epochs,
        mode='max')

    # Continuous eval callback
    evaluation = ContinuousEval(eval_frequency,
                                eval_files,
                                learning_rate,
                                job_dir,
                                pp_dir)

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, evaluation, tblog]

    X, y = model.generate_input(train_files, pp_dir)
    lstm_model.fit(X, y, epochs=num_epochs, callbacks=callbacks)

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if job_dir.startswith("gs://"):
        lstm_model.save(MODEL_NAME)
        copy_file_to_gcs(job_dir, MODEL_NAME)
    else:
        lstm_model.save(os.path.join(job_dir, MODEL_NAME))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(lstm_model, os.path.join(job_dir, 'export'))


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters needed for Cognites Hosting Environment to fill
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--pp-dir',
                        type=str,
                        help='GCS or local dir to write necessary preprocessing objects to')

    # Optional parameters for user to specify
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for training steps')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for evaluation steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--eval-frequency',
                        default=10,
                        help='Perform one evaluation per n epochs')
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=[30],
                        help='List of layer sizes in LSTM.')
    parser.add_argument('--eval-num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs during evaluation')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=30,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=5,
                        help='Checkpoint per n training epochs')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
