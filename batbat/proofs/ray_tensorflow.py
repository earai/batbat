import tensorflow as tf
import ray


def main():
    ray.init()

    @ray.remote
    def f(x):
        return (tf.constant(x) * tf.constant(x)).numpy()

    futures = [f.remote(i) for i in range(4)]
    print(ray.get(futures)) # [0, 1, 4, 9]

if __name__=="__main__":
    main()