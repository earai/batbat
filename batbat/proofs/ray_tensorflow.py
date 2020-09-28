import tensorflow as tf
import ray
from tensorflow_probability import distributions as tfd


def main():
    ray.init()

    @ray.remote
    def f(x):
        return (tfd.Normal(loc=x, scale=x).sample() * tf.constant(x,dtype=tf.float32)).numpy()

    futures = [f.remote(i) for i in range(4)]
    print(ray.get(futures)) # [0, 1, 4, 9]

if __name__=="__main__":
    main()