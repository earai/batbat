import tensorflow as tf

def test_eager():
    assert tf.executing_eagerly()