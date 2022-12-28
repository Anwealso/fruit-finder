"""
train.py

Alex Nicholson (45316207)
11/10/2022

Contains the source code for training, validating, testing and saving your model. The model is imported from “modules.py” and the data loader is imported from “dataset.py”. Losses and metrics are plotted throughout training.

"""


import dataset
import modules
import utils
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imageio.v2 as imageio
import object_detection.utils.config_util as config_util
from object_detection.builders import model_builder

# from object_detection.exporter import export_inference_graph
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

import random as random


# class VQVAETrainer(keras.models.Model):
#     """
#     A Custom Training Loop for the VQ-VAE Model

#         Attributes:
#             train_variance (ndarray): The input data for the model (input data in the form of variances?) ???
#             latent_dim (int): The number of latent dimensions the images are compressed down to (default=32)
#             num_embeddings (int): The number of codebook vectors in the embedding space (default=128)
#             vqvae (Keras Model): The custom VQ-VAE model
#             total_loss_tracker (Keras Metric): A tracker for the total loss performance of the model during training???
#             reconstruction_loss_tracker (Keras Metric): A tracker for the reconstruction loss performance of the model during training???
#             vqvae_loss_tracker (Keras Metric): A tracker for the VQ loss performance of the model during training???

#         Methods:
#             metrics(): Returns a list of metrics for the total_loss, reconstruction_loss, and vqvae_loss of the model
#             train_step(x): Trains the model for a single step using the given training sample/samples x???
#     """

#     def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
#         super(VQVAETrainer, self).__init__(**kwargs)
#         self.train_variance = train_variance
#         self.latent_dim = latent_dim
#         self.num_embeddings = num_embeddings

#         self.vqvae = modules.get_vqvae(self.latent_dim, self.num_embeddings)

#         self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
#         self.vqvae_loss_tracker = keras.metrics.Mean(name="vqvae_loss")

#         self.ssim_history = []

#     @property
#     def metrics(self):
#         """
#         Gets a list of metrics for current total_loss, reconstruction_loss, and vqvae_loss of the model
        
#             Returns:
#                 A list of metrics for total_loss, reconstruction_loss, and vqvae_loss
#         """

#         return [
#             self.total_loss_tracker,
#             self.reconstruction_loss_tracker,
#             self.vqvae_loss_tracker,
#         ]

#     def train_step(self, x):
#         """
#         Trains the model for a single step using the given training sample/samples x???

#             Parameters:
#                 x (Tensorflow Tensor???): The input training sample/samples (how big is a training step? how many samples?) ???

#             Returns:
#                 A dictionary of the model's training metrics with keys: "loss", "reconstruction_loss", and "vqvae_loss"
#         """

#         with tf.GradientTape() as tape:
#             # Outputs from the VQ-VAE.
#             reconstructions = self.vqvae(x)

#             # Calculate the losses.
#             reconstruction_loss = (
#                 tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
#             )
#             total_loss = reconstruction_loss + sum(self.vqvae.losses)

#         # Backpropagation.
#         grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

#         # Loss tracking.
#         self.total_loss_tracker.update_state(total_loss)
#         self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#         self.vqvae_loss_tracker.update_state(sum(self.vqvae.losses))

#         # Log results.
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#             "vqvae_loss": self.vqvae_loss_tracker.result(),
#         }


# class ProgressImagesCallback(keras.callbacks.Callback):
#     """
#     A custom callback for saving training progeress images
#     """

#     def __init__(self, train_data, validate_data):
#         self.train_data = train_data
#         self.validate_data = validate_data

#     def save_progress_image(self, epoch):
#         """
#         Saves progress images as we go throughout training

#             Parameters:
#                 epoch (int): The current training epoch
#         """

#         num_examples_to_generate = 16
#         idx = np.random.choice(len(self.train_data), num_examples_to_generate)
#         test_images = self.train_data[idx]
#         reconstructions_test = self.model.vqvae.predict(test_images)

#         fig = plt.figure(figsize=(16, 16))
#         for i in range(reconstructions_test.shape[0]):
#             plt.subplot(4, 4, i + 1)
#             plt.imshow(reconstructions_test[i, :, :, 0], cmap='gray')
#             plt.axis('off')

#         plt.savefig('out/image_at_epoch_{:04d}.png'.format(epoch+1))
#         plt.close()

#     def create_gif(self):
#         """
#         Show an animated gif of the progress throughout training
#         """

#         anim_file = 'out/vqvae_training_progression.gif'

#         with imageio.get_writer(anim_file, mode='I') as writer:
#             filenames = glob.glob('out/image*.png')
#             filenames = sorted(filenames)
#             for filename in filenames:
#                 image = imageio.imread(filename)
#                 writer.append_data(image)
#                 image = imageio.imread(filename)
#                 writer.append_data(image)

#     def on_epoch_end(self, epoch, logs=None):
#         self.save_progress_image(epoch)
        
#         similarity = utils.get_model_ssim(self.model.vqvae, self.validate_data)
#         self.model.ssim_history.append(similarity)
#         print(f"ssim: {similarity}")

#     def on_train_end(self, logs=None):
#         self.create_gif()


def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                        groundtruth_boxes_list,
                        groundtruth_classes_list):
        """A single training iteration.

        Args:
        image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
        groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
        groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
        A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                [detection_model.preprocess(image_tensor)[0]
                for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)

            # print("=================================================================")
            # print(f"model: {model}\n")
            # print(f"preprocessed_images: {preprocessed_images}\n")
            # print(f"shapes: {shapes}\n")
            # print(f"prediction_dict: {prediction_dict}\n")

            losses_dict = model.loss(prediction_dict, shapes)
            # print(f"losses_dict: {losses_dict}\n")            

            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn

@tf.function # uncomment this decorator if you want to run inference eagerly
def detect(detection_model, input_tensor):
    """Run detection on an input image.

    Args:
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.

    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    return detection_model.postprocess(prediction_dict, shapes)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                HYPERPARAMETERS                               #
    # ---------------------------------------------------------------------------- #
    DATASET_PATH = ".\\data\\totoro\\"
    MODEL_PATH = ".\\models\\ssd_mobilenet_v2_fpnlite_640x640_totoro\\"
    OUTPUT_DIRECTORY = ".\\exported-models\\my_model_pre" # the dir into which the trained model will be saved

    MAX_TRAINING_EXAMPLES = None

    TRAINING_EPOCHS = 5
    BATCH_SIZE = 128

    NUM_LATENT_DIMS = 16
    NUM_EMBEDDINGS = 128

    EXAMPLES_TO_SHOW = 10


    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    print("\nLoading dataset...")
    (train_dataset, validate_dataset, test_dataset) = dataset.load_dataset(DATASET_PATH, max_images=MAX_TRAINING_EXAMPLES, verbose=2)
    
    # ---------------------------------------------------------------------------- #
    #                            LOAD PRE-TRAINED MODEL                            #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved model from file
    print('\nBuilding model and restoring weights for fine-tuning...', flush=True)

    tf.keras.backend.clear_session()

    num_classes = 1 # TODO: Nodify this to pull tghe desired number of classes from our label_map.pbtxt
    config_path = MODEL_PATH + 'pipeline.config'
    checkpoint_dir = MODEL_PATH + 'checkpoint\\'
    checkpoint_path = MODEL_PATH + 'checkpoint\\ckpt-0'

    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(config_path)
    configs['model'].ssd.num_classes = num_classes
    configs['model'].ssd.freeze_batchnorm = True
    configs['train_config'].batch_size = 4
    configs['train_config'].fine_tune_checkpoint = checkpoint_path # Path to checkpoint of pre-trained model
    configs['train_config'].fine_tune_checkpoint_type = "detection" # Set this to "detection" since we want to be training the full detection model
    configs['train_config'].use_bfloat16 = False # Set this to false if you are not training on a TPU
    configs['train_input_config'].label_map_path = DATASET_PATH + "label_map.pbtxt" # Path to label map file
    # configs['train_input_config'].tf_record_input_reader.input_path = "annotations/train.record" # Path to training TFRecord file
    configs['train_config'].use_bfloat16 = False # Set this to false if you are not training on a TPU
    # configs['eval_config'].metrics_set = "coco_detection_metrics"
    # configs['eval_config'].use_moving_averages = False
    configs['eval_input_config'].label_map_path = DATASET_PATH + "label_map.pbtxt" # Path to label map file
    # configs['eval_input_config'].tf_record_input_reader .input_path = "annotations/test.record" # Path to testing TFRecord

    pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_proto, MODEL_PATH)

    print(f"Model+Training Configurations:\n{configs}\n")
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=True)


    # ---------------------------------------------------------------------------- #
    #                      MODIFY MODEL TO FIT DESIRED CLASSES                     #
    # ---------------------------------------------------------------------------- #
    # Remove the final classification layer and replace it with a basic randomly 
    # initialised classifier layer that classifies for our classes
    print("\nModifying model to fit desired classes ...")

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
    fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')

    print(f"Model+Training Configurations 2.0:\n{configs}\n")

    # ---------------------------------------------------------------------------- #
    #                                 RUN TRAINING                                 #
    # ---------------------------------------------------------------------------- #
    # Finetune the modified pre-trained model on the smaller dataset containing 
    # examples of our classes
    
    print('\nStarting fine-tuning...', flush=True)

    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could
    # fit more examples in memory if we wanted to.
    batch_size = 4
    learning_rate = 0.01
    num_batches = 100
    # num_batches = 10

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    # Set up forward + backward pass for a single train step.
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune)

    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_dataset["images"])))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        # TODO: Do data augmentation
        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping
        # and random cropping :)
        gt_boxes_list = [train_dataset["labels"][key] for key in example_keys]

        gt_classes_list = [train_dataset["gt_classes_one_hot_tensors"][key] for key in example_keys]
        image_tensors = [train_dataset["train_image_tensors"][key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        if idx % 10 == 0:
            # Print out smoe training metrics
            print('batch ' + str(idx) + ' of ' + str(num_batches)
            + ', loss=' +  str(total_loss.numpy()), flush=True)
            # Save training checkpoint
            ckpt = tf.train.Checkpoint(model=detection_model)
            exported_checkpoint_manager = tf.train.CheckpointManager(
                ckpt, checkpoint_dir, max_to_keep=3)
            exported_checkpoint_manager.save(checkpoint_number=idx//10)

    print('Done fine-tuning!')

    # TODO: Add tensorboard plotting throughout training so I can see the loss and convergence characteristics
    # Run training, plotting losses and metrics throughout
    # history = vqvae_trainer.fit(train_data, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, callbacks=[ProgressImagesCallback(train_data, validate_data)])

    # ---------------------------------------------------------------------------- #
    #                           EXPORT THE TRAINED MODEL                           #
    # ---------------------------------------------------------------------------- #
    print('Exporting model...')

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(MODEL_PATH + "pipeline.config", 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge("", pipeline_config)

    exporter_lib_v2.export_inference_graph(
        "image_tensor", 
        pipeline_config, 
        MODEL_PATH + "checkpoint",
        OUTPUT_DIRECTORY)

    # model.save('path/to/location')


    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the model training curves
    # utils.plot_training_metrics(history)
    # utils.plot_ssim_history(vqvae_trainer.ssim_history)
    
    # Visualise output generations from the finished model
    # utils.show_reconstruction_examples(trained_vqvae_model, validate_data, EXAMPLES_TO_SHOW)


    # -------------------------- Setup the test dataset -------------------------- #
    # TODO: Rework this to pull the test images straight out of the loaded dataset
    test_image_dir = DATASET_PATH + 'images\\test\\'
    test_images_list = glob.glob(os.path.join(test_image_dir, "*"))
    test_images_np = []
    for image_path in test_images_list:
        # image_path = os.path.join(test_image_dir, 'test' + str(i) + '.jpg')
        test_images_np.append(np.expand_dims(
            utils.load_image_into_numpy_array(image_path), axis=0))

    # Note that the first frame will trigger tracing of the tf.function, which will
    # take some time, after which inference should be fast.
    label_id_offset = 1
    for i in range(len(test_images_np)):
        input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
        detections = detect(detection_model, input_tensor)

        out_filename = test_images_list[i].replace("images\\", "out\\")

        print(out_filename)
        utils.plot_detections(
            test_images_np[i][0],
            detections['detection_boxes'].numpy(),
            detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
            detections['detection_scores'][0].numpy(),
            figsize=(15, 20), image_name=out_filename, min_score=0.2)
