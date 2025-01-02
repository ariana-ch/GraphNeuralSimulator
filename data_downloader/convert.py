import os
import json
import pickle
import argparse
import functools
import numpy as np
import tensorflow.compat.v1 as tf

# Feature descriptions for TFRecord parsing
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string
)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = [np.frombuffer(el.numpy(), dtype=encoded_dtype) for el in x]
    return tf.convert_to_tensor(np.array(out))


def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample."""
    feature_description = (_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
                           if 'context_mean' in metadata else _FEATURE_DESCRIPTION)

    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description
    )

    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in']
        )
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out']
        )

    # Reshape positions to correct dimensions
    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]
    parsed_features['position'] = tf.reshape(parsed_features['position'], position_shape)

    if 'context_mean' in metadata:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = tf.reshape(
            parsed_features['step_context'],
            [metadata['sequence_length'] + 1, context_feat_len]
        )

    context['particle_type'] = tf.py_function(
        functools.partial(convert_to_tensor, encoded_dtype=np.int64),
        inp=[context['particle_type'].values],
        Tout=[tf.int64]
    )
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])

    return context, parsed_features


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        return json.load(f)


def input_fn(data_path, split):
    metadata = _read_metadata(data_path)
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        parse_serialized_simulation_example, metadata=metadata
    ))
    return ds


def convert_dataset(data_path: str = './tmp/WaterDropSample'):
    """Converts TFRecord dataset to .pkl files."""
    print(f'Converting dataset {data_path.split(os.sep)[-1]}')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        metadata = _read_metadata(data_path)

        save_dir = os.path.join('./data', data_path.split(os.sep)[-1])
        os.makedirs(save_dir, exist_ok=True)
        json.dump(metadata, open(os.path.join(save_dir, 'metadata.json'), 'w'))

        for split in ['train', 'valid', 'test']:
            ds = input_fn(data_path, split=split)
            split_dir = os.path.join(save_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            iterator = tf.data.make_one_shot_iterator(ds)
            data = iterator.get_next()

            i = 0
            while True:
                try:
                    traj = sess.run(data)
                    traj = {
                        'particle_type': traj[0]['particle_type'],
                        'position': traj[1]['position']
                    }
                    with open(os.path.join(split_dir, f'{i}.pkl'), 'wb') as f:
                        pickle.dump(traj, f)
                    i += 1

                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    convert_dataset('/tmp/WaterRamps')
    parser = argparse.ArgumentParser(description='Convert TFRecord datasets to .pkl')
    parser.add_argument('--data_path', type=str, default='./tmp/WaterDropSample',
                        help='Path to the TFRecord dataset')
    args = parser.parse_args()

    convert_dataset(args.data_path)
