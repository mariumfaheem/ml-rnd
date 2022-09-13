from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
tfds.disable_progress_bar()
import logging
from datetime import datetime, timezone
logger = tf.get_logger()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO)
print('Tensorflow-version: {0}'.format(tf.__version__))

import os
import argparse
import json

# parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-batch-size',
                        type=int,
                        default='8',
                        help='BATCH_SIZE for training.')
    parser.add_argument('--tf-epochs',
                        type=int,
                        default=2,
                        help='The number of training steps to perform.')
    parser.add_argument('--tf-dropout',
                        type=float,
                        default=0.2,
                        help='Dropout')
    parser.add_argument('--tf-num-layers',
                        type=int,
                        default=1,
                        help='The number of training steps to perform.')
    parser.add_argument('--tf-num-dims',
                        type=int,
                        default=64,
                        help='dims')
    parser.add_argument('--tf-learning-rate',
                        type=float,
                        default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--tf-optimizer',
                        type=str,
                        default="adam",
                        help='optimizer: adam, sgd, ftrl')


    args = parser.parse_known_args()[0]
    return args


# prepare data
def prepare_data(batch_size=64, shuffle_size=1000):

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    # Split the training set into 70% 15% and 15% for training test and validation
    (train_data, validation_data, test_data),info = tfds.load(name="fashion_mnist",
                                                              split=('train[:70%]', 'train[70%:85%]', 'train[85%:]'),
                                                              as_supervised=True, with_info=True)


    print("Training data count : ", int(info.splits['train'].num_examples * 0.8))
    print("Validation data count : ", int(info.splits['train'].num_examples * 0.2))
    print("Test data count : ", int(info.splits['test'].num_examples))


    # create dataset to be used for training process
    train_dataset = train_data.map(scale).shuffle(shuffle_size).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = validation_data.map(scale).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_data.map(scale).batch(batch_size)

    return train_dataset, val_dataset, test_dataset


def optimizers(TF_LEARNING_RATE):
    if args.tf_optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(lr=TF_LEARNING_RATE)
    elif args.tf_optimizer == "ftrl":
        opt = tf.keras.optimizers.Ftrl(lr=TF_LEARNING_RATE)
    else:
        opt = tf.keras.optimizers.Adam(lr=TF_LEARNING_RATE)



def build_model():

    # parse arguments
    args = parse_arguments()

    TF_NUM_LAYERS = int(args.tf_num_layers)
    TF_NUM_DIMS = int(args.tf_num_dims)
    TF_DROPOUT = float(args.tf_dropout)
    TF_LEARNING_RATE = float(args.tf_learning_rate)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1), name='x'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    for i in range(TF_NUM_LAYERS):
        model.add(tf.keras.layers.Dense(TF_NUM_DIMS, activation='relu'))
        model.add(tf.keras.layers.Dropout(TF_DROPOUT))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    opt = optimizers(TF_LEARNING_RATE)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )
    return model


# callbacks
def get_callbacks():
    # callbacks
    # checkpoint directory
    checkpointdir = '/tmp/model-ckpt'

    class customLog(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            local_time = datetime.now(timezone.utc).astimezone().isoformat()
            logging.info("\n{} Train-epoch={} Train-accuracy={:.4f} Train-loss={:.4f} Validation-accuracy={:.4f} Validation-loss={:.4f}".format(local_time, epoch + 1, logs['accuracy'], logs['loss'], logs['val_accuracy'], logs['val_loss']))

    callbacks = [
        #tf.keras.callbacks.TensorBoard(logdir),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpointdir),
        customLog()
    ]
    return callbacks


def main():

    # parse arguments
    args = parse_arguments()

    tf_config = os.environ.get('TF_CONFIG', '{}')
    logging.info("TF_CONFIG %s", tf_config)
    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')
    logging.info("cluster=%s job_name=%s task_index=%s", cluster, job_name,
                 task_index)

    is_chief = False
    if not job_name or job_name.lower() in ["chief", "master"]:
        is_chief = True
        logging.info("Will export model")
    else:
        logging.info("Will not export model")

    # multi-worker mirrored strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    logging.info("Number of devices: {0}".format(strategy.num_replicas_in_sync))


    # build keras model
    with strategy.scope():
        # Data extraction and processing
        # set variables
        BUFFER_SIZE = 10000
        BATCH_SIZE = int(args.tf_batch_size)
        BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
        train_dataset, val_dataset, test_dataset = prepare_data(batch_size=BATCH_SIZE, shuffle_size=BUFFER_SIZE)


        # build and compile model
        learning_rate = float(args.tf_learning_rate)
        logging.info("learning rate : {0}".format(learning_rate))
        model = build_model()
        model.summary()

        # training and evaluation
        logging.info("Training starting...")
        TF_STEPS_PER_EPOCHS = 5
        #TF_STEPS_PER_EPOCHS = int(np.ceil(60000 / float(BATCH_SIZE)))

        # train model
        EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        model.fit(train_dataset,
                  epochs=int(args.tf_epochs),
                  steps_per_epoch=TF_STEPS_PER_EPOCHS,
                  validation_data=val_dataset,
                  validation_steps=1,
                  callbacks=[get_callbacks(), EarlyStopping])
        logging.info("Training completed.")


        # model save
        if is_chief:
            # save the model
            model.save("model.h5")
            logging.info("model saved.")


    # load the model
    model_loaded = tf.keras.models.load_model('model.h5')

    # evaluate model on the test dataset
    loss, accuracy = model_loaded.evaluate(test_dataset)
    print("\nfinal evaluation : Test-loss={0:.4f}, Test-accuracy={1:.4f}".format(loss, accuracy))
    print("process completed.")
    # successful completion
    exit(0)


if __name__ == "__main__":
    main()