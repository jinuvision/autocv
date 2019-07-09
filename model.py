# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon, Heung-Chang Lee, Do-Guk Kim


import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
from urllib.request import urlopen
from tensorflow.python.ops import array_ops
import mobilenet_v2

tf.logging.set_verbosity(tf.logging.ERROR)

class Model(object):
  """Construct a model with 3D CNN for classification."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False
    self.metadata = metadata

    # Get the output dimension, i.e. number of classes
    self.output_dim = self.metadata.get_output_size()
    # Set batch size (for both training and testing)
    self.batch_size = 50

    # Attributes for preprocessing
    self.default_image_size = (112,112)
    self.default_num_frames = 10
    self.default_shuffle_buffer = 100

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.estimated_time_per_step = None
    self.total_test_time = 0
    self.cumulated_num_tests = 0
    self.train_cnt = 0
    self.estimated_time_test = None
    # Critical number for early stopping
    self.num_epochs_we_want_to_train = 100
    self.image_size = metadata.get_matrix_size()
    if self.image_size[0] > 224:
        self.image_size = (224,224)
    if self.image_size[0] < 0 or self.image_size[1] < 0:
        self.image_size = self.default_image_size
    
    self.input_size = min(self.image_size[0], self.image_size[1])
    
    # Get model function from class method below
    model_fn = self.model_fn
    # Classifier using model_fn
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="./submission/",
                      vars_to_warm_start="(?=.*Mobilenet)(?=^(?!.*Logits)).*")
    tensor_shape = metadata.get_tensor_shape()
    if tensor_shape[3] == 3:
      self.classifier = tf.estimator.Estimator(model_fn=model_fn, warm_start_from=ws)
    else:
      self.classifier = tf.estimator.Estimator(model_fn=model_fn)
    
  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """       
    #print (self.image_size)    
    # Get number of steps to train according to some strategy
    steps_to_train = self.get_steps_to_train(remaining_time_budget)
    
    
    # Count examples on training set        
    if self.train_cnt == 0 and (self.image_size[0] < 0 or self.image_size[1] < 0):      
      iterator = dataset.make_one_shot_iterator()
      example, labels = iterator.get_next()
      need_simple_arch = False
      with tf.Session() as sess:
        while True:
          try:
            curr = sess.run(example)
            curr_shape = curr.shape
            if min(curr_shape[1], curr_shape[2]) <= 28:
              need_simple_arch = True
              break                        
          except tf.errors.OutOfRangeError:
            break
      if need_simple_arch:
        self.input_size = min(curr_shape[1], curr_shape[2])
        self.classifier = tf.estimator.Estimator(model_fn=self.model_fn)
    
    self.num_examples_train = 0
    if steps_to_train <= 0:
      logger.info("Not enough time remaining for training. " +
            "Estimated time for training per step: {:.2f}, "\
            .format(self.estimated_time_per_step) +
            "but remaining time budget is: {:.2f}. "\
            .format(remaining_time_budget) +
            "Skipping...")
      self.done_training = True        
    else:
      msg_est = ""
      if self.estimated_time_per_step:
        msg_est = "estimated time for this: {:.2f} sec."\
                  .format(steps_to_train * self.estimated_time_per_step)
      logger.info("Begin training for another {} steps...{}"\
                  .format(steps_to_train, msg_est))
      
      # Prepare input function for training
      train_input_fn = lambda: self.input_function(dataset, is_training=True)
      
      # Start training
      train_start = time.time()
      self.classifier.train(input_fn=train_input_fn, steps=steps_to_train)        
      train_end = time.time()

      # Update for time budget managing      
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      logger.info("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
            "Now total steps trained: {}. ".format(self.cumulated_num_steps) +\
            "Total time used for training: {:.2f} sec. ".format(self.total_train_time) +\
            "Current estimated time per step: {:.2e} sec.".format(self.estimated_time_per_step))
      self.train_cnt += 1
      

  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
    test_begin = time.time()
    logger.info("Begin testing... ")

    # Prepare input function for testing
    test_input_fn = lambda: self.input_function(dataset, is_training=False)

    # Start testing (i.e. making prediction on test set)
    test_results = self.classifier.predict(input_fn=test_input_fn)

    predictions = [x['probabilities'] for x in test_results]
    predictions = np.array(predictions)
    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    self.total_test_time += test_duration
    self.cumulated_num_tests += 1
    self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
    logger.info("Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
          "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
          "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  # Model functions that contain info on neural network architectures
  # Several model functions are to be implemented, for different domains
  
  def model_fn(self, features, labels, mode):
    """Auto-Scaling 3D CNN model.

    For more information on how to write a model function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
    """
    input_layer = features

    # Replace missing values by 0
    hidden_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)
    

    if self.input_size > 28:
        hidden_layer = tf.squeeze(hidden_layer, axis=[1])
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
            logits, endpoints = mobilenet_v2.mobilenet(hidden_layer, self.input_size)
        hidden_layer = tf.contrib.layers.conv2d(
                inputs=endpoints['feature_maps'], num_outputs=1280, kernel_size=1, stride=1, activation_fn=None)
        hidden_layer = tf.reduce_mean(input_tensor=hidden_layer, axis=[1, 2])
    else:
        REASONABLE_NUM_ENTRIES = 1000
        num_filters = 32 # The number of filters is fixed
        while True:
          shape = hidden_layer.shape
          kernel_size = [min(3, shape[1]), min(3, shape[2]), min(3, shape[3])]
          hidden_layer = tf.layers.conv3d(inputs=hidden_layer,
                                          filters=num_filters,
                                          kernel_size=kernel_size)
          kernel_size = [min(1, shape[1]), min(1, shape[2]), min(1, shape[3])]
          hidden_layer = tf.layers.conv3d(inputs=hidden_layer,
                                          filters=num_filters,
                                          kernel_size=kernel_size)    
          pool_size = [min(2, shape[1]), min(2, shape[2]), min(2, shape[3])]
          hidden_layer= tf.layers.max_pooling3d(inputs=hidden_layer,
                                                pool_size=pool_size,
                                                strides=pool_size,
                                                padding='valid',
                                                data_format='channels_last')
          if get_num_entries(hidden_layer) < REASONABLE_NUM_ENTRIES:
            break 
        hidden_layer = tf.layers.flatten(hidden_layer)
        
    hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                   units=256, activation=tf.nn.relu)
    hidden_layer = tf.layers.dropout(
        inputs=hidden_layer, rate=0.5,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=hidden_layer, units=self.output_dim)
    sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # "classes": binary_predictions,
      # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": sigmoid_tensor
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # For multi-label classification, a correct loss is sigmoid cross entropy
    loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    #loss = focal_loss(prediction_tensor=logits, target_tensor=labels)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(0.001)      
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


  def input_function(self, dataset, is_training):
    """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
    """
    
    dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))

    if is_training:
      # Shuffle input examples
      dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
      # Convert to RepeatDataset to train for several epochs
      dataset = dataset.repeat()

    # Set batch size
    dataset = dataset.batch(batch_size=self.batch_size)

    iterator = dataset.make_one_shot_iterator()
    example, labels = iterator.get_next()
    return example, labels

  def preprocess_tensor_4d(self, tensor_4d):
    """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
    non-fixed). The output tensor wil have fixed, known shape.

    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels]
          where some dimensions might be `None`.
    Returns:
      A 4-D Tensor with fixed, known shape.
    """
    tensor_4d_shape = tensor_4d.shape
    logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

    if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
      num_frames = tensor_4d_shape[0]
    else:
      num_frames = self.default_num_frames
    
    '''
    if self.output_dim in [6, 10]:
      new_row_count = 28
    elif tensor_4d_shape[1] > 0:
#       if tensor_4d_shape[1] > 224:
#         new_row_count = 224
#       else:
        #new_row_count = tensor_4d_shape[1]
        new_row_count = self.default_image_size[0]
    else:
      new_row_count=self.default_image_size[0]
    if self.output_dim in [6, 10]:
      new_col_count = 28
#     elif self.output_dim in [3, 26]:
#       new_col_count = 56
    elif tensor_4d_shape[2] > 0:
#       if tensor_4d_shape[2] > 224:
#         new_col_count = 224
#       else:
        #new_col_count = tensor_4d_shape[2]
        new_col_count = self.default_image_size[1]
    else:
      new_col_count=self.default_image_size[1]
    '''
    
    new_row_count, new_col_count = self.image_size[0], self.image_size[1]
    
    if not tensor_4d_shape[0] > 0:
      logger.info("Detected that examples have variable sequence_size, will " +
                "randomly crop a sequence with num_frames = " +
                "{}".format(num_frames))
      tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
    if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
      logger.info("Detected that examples have variable space size, will " +
                "resize space axes to (new_row_count, new_col_count) = " +
                "{}".format((new_row_count, new_col_count)))
    tensor_4d = resize_space_axes(tensor_4d,
                                  new_row_count=new_row_count,
                                  new_col_count=new_col_count)
    logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))    
    
    return tensor_4d

  def get_steps_to_train(self, remaining_time_budget):
    """Get number of steps for training according to `remaining_time_budget`.

    The strategy is:
      1. If no training is done before, train for 10 steps (ten batches);
      2. Otherwise, estimate training time per step and time needed for test,
         then compare to remaining time budget to compute a potential maximum
         number of steps (max_steps) that can be trained within time budget;
      3. Choose a number (steps_to_train) between 0 and max_steps and train for
         this many steps. Double it each time.
    """
    if not remaining_time_budget: # This is never true in the competition anyway
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    image_max_len = max(self.image_size[0], self.image_size[1])
    adaptive_batch = 100
    if not self.estimated_time_per_step:
      steps_to_train = adaptive_batch
    else:
      if self.estimated_time_test:
        tentative_estimated_time_test = self.estimated_time_test
      else:
        tentative_estimated_time_test = 50 # conservative estimation for test
      max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
      max_steps = max(max_steps, 1)
      if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
        steps_to_train = int(adaptive_batch * (1.5 ** self.cumulated_num_tests)) # Double steps_to_train after each test
      else:
        steps_to_train = 0
      
    return steps_to_train

  def age(self):
    return time.time() - self.birthday

  def choose_to_stop_early(self):
    """The criterion to stop further training (thus finish train/predict
    process).
    """
    batch_size = self.batch_size
    num_examples = self.num_examples_train
    num_epochs = self.cumulated_num_steps * batch_size / num_examples
    logger.info("Model already trained for {} epochs.".format(num_epochs))
    return num_epochs > self.num_epochs_we_want_to_train # Train for at least certain number of epochs then stop

def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
  """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
  labels = tf.cast(labels, dtype=tf.float32)
  relu_logits = tf.nn.relu(logits)
  exp_logits = tf.exp(- tf.abs(logits))
  sigmoid_logits = tf.log(1 + exp_logits)
  element_wise_xent = relu_logits - labels * logits + sigmoid_logits
        
  return tf.reduce_sum(element_wise_xent)

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.5, gamma=5):  
  sigmoid_p = tf.nn.sigmoid(prediction_tensor)
  zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
  pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
  neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
  per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

  return tf.reduce_sum(per_entry_cross_ent)

def get_num_entries(tensor):
  """Return number of entries for a TensorFlow tensor.

  Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
  Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
  """
  tensor_shape = tensor.shape
  assert(len(tensor_shape) > 1)
  num_entries  = 1
  for i in tensor_shape[1:]:
    num_entries *= int(i)
  return num_entries

def crop_time_axis(tensor_4d, num_frames, begin_index=None):
  """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[1], 0)
  padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[0] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1, -1])

  return sliced_tensor

def resize_space_axes(tensor_4d, new_row_count, new_col_count):
  """Given a 4-D tensor, resize space axes to have target size.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
  resized_images = tf.image.resize_images(tensor_4d,
                                          size=(new_row_count, new_col_count))
  return resized_images

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')

