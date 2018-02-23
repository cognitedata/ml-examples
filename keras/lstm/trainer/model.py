import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import layers, models
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from trainer import processing


def model_fn(input_shape, hidden_layers):
    """Create a Keras Sequential model with layers."""
    model = models.Sequential()

    for num_units in hidden_layers[:-1]:
        model.add(layers.LSTM(num_units, return_sequences=True, input_shape=input_shape))

    model.add(layers.LSTM(hidden_layers[-1], input_shape=input_shape))

    # Add a dense final layer with sigmoid function
    model.add(layers.Dense(1))
    model = compile_model(model)
    return model


def compile_model(model):
    model.compile(loss='mse', optimizer='adam')
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'output': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def generate_input(input_files, pp_dir):
    df = pd.read_csv(tf.gfile.Open(input_files[0]))
    X, y = processing.preprocess(df, pp_dir)
    return X, y


# def generator_input(input_file, chunk_size):
#     """Generator function to produce features and labels
#        needed by keras fit_generator.
#     """
#     def generator():
#         while True:
#             input_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
#                                        names=CSV_COLUMNS,
#                                        chunksize=chunk_size,
#                                        na_values=" ?")
#             for input_data in input_reader:
#                 input_data = input_data.dropna()
#                 label = input_data.pop(LABEL_COLUMN)
#                 n_rows = input_data.shape[0]
#                 n_cols = input_data.shape[1]
#                 input_data = input_data.values.reshape(n_rows, 2, n_cols / 2)
#                 label = label.values.reshape(-1, 1)
#                 yield input_data, label
#
#     return generator()
