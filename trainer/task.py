"""
Creates a Model and executes a train and evaluate routine locally, or on MLEngine.
Arguments passed by parseline arguments are grouped on project, training and hyperparameter 
PROJECT
    --project car-prediction-220612 \
    --temp_location gs://car-prediction-220612/temp \
    --data_folder path to the training, validation and test data
    --file_name_train base name of training files
    --file_name_test base name of test files
    --output_folder path to folder to output checkpoints, model 
    --temp_file path to folder for staging
TRAINING
    --num_epochs number of times the wholedataset is processed for training
    --batch_size size of batch during training, serving is one
HYPERPARAMETERS
    --step_size size of the gradient descent step in the optimizer
    --hidden_layers number of hidden layers in the networks
    --nodes number of nodes in a hidden layer
    --activation_function 'sig', 'rel'
    
COMMANDS TO STORE ENVIRONEMNT VARIABLES
    now=$(date +"%Y%m%d_%H%M%S")
    JOB_NAME="car_dnnreg_$now"
    MAIN_TRAINER_MODULE="trainer.task"
    TRAIN_DATA="train_data"
    TEST_DATA="val_data"
    
COMMANDS TO TRAIN MLENGINE LOCALLY
    gcloud ml-engine local train \
    --module-name $MAIN_TRAINER_MODULE \
    --package-path trainer \
    --job-dir output/$JOB_NAME \
    -- \
    --file_name_train $TRAIN_DATA \
    --file_name_test $TEST_DATA \
    --data_folder output \
    --max_train_steps 4000 \
    --batch_size 100

COMMANDS TO TRAIN ON MLENGINE 
gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket gs://car-prediction-220612/ \
    --module-name trainer.task \
    --package-path trainer \
    --job-dir gs://car-prediction-220612/jobs/$JOB_NAME \
    --project car-prediction-220612 \
    --region europe-west1 \
    --config config.yaml \
    -- \
    --file_name_train $TRAIN_DATA \
    --file_name_test $TEST_DATA \
    --data_folder gs://car-prediction-220612/output \
    --max_train_steps 4000 \
    --batch_size 128

COMMAND TO PREDICT LOCALLY
    MODEL=output/car_dnnreg_20190110_132224/1547123891 
    INPUT_DATA_FILE="test_data_online.json"
    gcloud ml-engine local predict \
    --model-dir=$MODEL \
    --json-instances=test_data.json

COMMAND TO PREDICT ONLINE 
    MODEL_NAME="carpredictor"
    INPUT_DATA_FILE="test_data.json"
    VERSION_NAME="v3"

    gcloud ml-engine predict --model $MODEL_NAME  \
        --version $VERSION_NAME \
        --json-instances $INPUT_DATA_FILE

By: Eric van der Knaap
"""

from __future__ import absolute_import
from __future__ import division

from os.path import dirname, realpath, sep, pardir
import sys

sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "input_metadata")

import os
import time
import datetime
import argparse

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from input_metadata import CATEGORICAL_FEATURE_KEYS
from input_metadata import BUCKETIZED_FEATURES_KEYS
from input_metadata import HASHED_FEATURE_KEYS
from input_metadata import MILEAGE_BUCKETS
from input_metadata import BUCKETIZED_FEATURES
from input_metadata import NUMERIC_FEATURE_KEYS
from input_metadata import CODING_FEATURE_KEYS
from input_metadata import RAW_DATA_METADATA
from input_metadata import LABEL_KEY

def _make_training_input_function(tf_transform_output, transformed_examples, batch, num_epochs):
    """ Create an input function for training 
    Args:
        tf_transform_output: wrapper around the transform output
        transformed_examples: location and base name of the training dataset 
        batch: batch size for training
        num_epochs: number of epochs in training *(unless early stop criterion or max_steps are reached) 
    Output:
        input_fn:
    """

    def input_fn():
        """ Returns a batch of transformed features & labels for training
        Args:
        Output:
            transformed_features:   shuffled batch of features
            transformed_labels:     shuffled batch of labels 
        
        """
        dataset = tf.contrib.data.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=True,
        )
        dataset = dataset.repeat(num_epochs)
        
        # Get a batch of transformed features, including labels
        transformed_features = dataset.make_one_shot_iterator().get_next()

        transformed_labels = transformed_features.pop(LABEL_KEY[0])

        return transformed_features, transformed_labels
    return input_fn

def _make_serving_input_function(tf_transform_output):
    """ Creates a serving input function during production
    Args:
        tf_transform_output: wrapper around transformed 
    Output:
        serving functions that feeds our model 
    """

    # Create the input metadata from for the input data 
    # This is a dictionary, defining for each feature, the variable type and 
    # If it is of fixed length or not. Variable length example, is the occurance of one or more 
    # object in an image with bounding boxes. 
    raw_feature_spec = RAW_DATA_METADATA.schema.as_feature_spec()
    raw_feature_spec.pop(LABEL_KEY[0])
        
    def serving_input_fn():
        """ Accepts inference requests at serve time in JSON format & passes as tensor to the model  
            Output: the serving function 
        """
        
        # Create placeholder for incoming data. Note we reuse the data format of the trainig data schema
        # E.g., Make & Model of the car are string, Milage and Year are floats
        feature_placeholders = dict(
            [(key, tf.placeholder(tf.string, None)) for key in CATEGORICAL_FEATURE_KEYS] +
            [(key, tf.placeholder(tf.float32, None)) for key in NUMERIC_FEATURE_KEYS] 
        )

        # We don't have a label at serve time
        feature_placeholders.pop(LABEL_KEY[0])

        # Use TFTransform to apply the same preprocessing transformations. 
        features = tf_transform_output.transform_raw_features(feature_placeholders)

        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)  
        

    # Return the preprocesses sample

    return serving_input_fn

def get_feature_columns(tf_transform_output):
    """ Returns FeatureColumns object, the model expects
    """

    numeric_columns = [tf.feature_column.numeric_column('Mileage'),
                        tf.feature_column.numeric_column('Year')]

    # During preprocessing we create files which hold the vocabularies
    # i.e., for make, model and City. We first the string to an integer index
    # then turn it into a one hot encoding using indicator column
    categorical_columns = [tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_file( 
            key=key,
            vocabulary_file=tf_transform_output.vocabulary_file_by_name(key)))
    for key in ['Make','Model']] 

    # The input of the bucketized columns are already in integer format, one for bucket it belongs to
    # To turn this into one hot encoding, 
    # bucketized_columns  = [tf.feature_column.indicator_column(
    #                         tf.feature_column.categorical_column_with_identity(
    #                             key = key, 
    #                             num_buckets = 
    #                                 len(BUCKETIZED_FEATURES[key])+1)) 
    #                     for key in BUCKETIZED_FEATURES_KEYS]
    # hash_col = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity('Model', num_buckets=100))]
    return categorical_columns + numeric_columns

def train_and_evaluate(data_dir, log_dir, file_name_train, file_name_test, train_batch, eval_batch, max_train_steps, arguments):
    """ Train and evaluate the model 
        Args:
            data_dir: path that contains the data. 
            log_dir:  path to folder that will contain checkpoints, exported models 
            train_batch: size of the training batch
            eval_batch: size of the test batch (default 1)
    """
    #import tensorflow_transform as tft
    # Wrapper around TFTransform, on disk in a bucket
    tf_transform_output = tft.TFTransformOutput(data_dir)

    # Three lines of code, of the actual classifier
    # model = tf.estimator.LinearRegressor(
    #     feature_columns=get_feature_columns(tf_transform_output),
    #     model_dir=log_dir,
    #     optimizer = tf.train.FtrlOptimizer(
    #                 learning_rate=arguments.step_size,
    #                 l1_regularization_strength=0.001))

    hidden_units = [max(2, int(arguments.first_hidden_units*arguments.scale_factor**i))
       for i in range(arguments.hidden_layers)]

    #Create a DNNLinear regressor instead 
    model = tf.estimator.DNNRegressor(
        feature_columns = get_feature_columns(tf_transform_output),
        hidden_units = hidden_units,
        model_dir=log_dir)
       
    # Create the input function, which loads TFRecords called 'tf_records_train'
    train_input_fn = _make_training_input_function(
        tf_transform_output,
        os.path.join(data_dir,file_name_train + '*'),
        batch = train_batch,
        num_epochs = arguments.num_epochs
    )

    # Create the input function, which loads TFRecord called 'tf_records_test'
    eval_input_fn = _make_training_input_function(
        tf_transform_output,
        os.path.join(data_dir,file_name_test + '*'),
        batch = eval_batch,
        num_epochs = arguments.num_epochs
    )

    # Add specifications for training
    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps = arguments.max_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn)

    # get the performance metric from the helper function. In this case it is the root mean square error
    # between the predice price, and the prediced car price
    model = tf.contrib.estimator.add_metrics(model, add_metric)
  
    # Train and evaluate te model
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # Create serving function
    serving_input_fn = _make_serving_input_function(tf_transform_output)
    model.export_saved_model(log_dir,serving_input_fn)

    return 

def add_metric(labels, predictions):
    pred_values = predictions['predictions']
    metric = {'RMSELOSS': tf.metrics.root_mean_squared_error(labels, pred_values)}
    
    return metric

def temp_filepath(name=''):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    st = name + st
    return st

def main():
    
    # Create an argument parser to pass training options like
    # location of data, name of the data, batch size for training , serving
    # Maybe later some hyperparameter stuff
    # Path to store 
    
    parser = argparse.ArgumentParser()
    
    # PROJECT ARGUMENTS
    parser.add_argument('--project',
                        type = str,
                        default = 'car-prediction-220612',
                        required = False)
    parser.add_argument('--temp_location',
                        type = str,
                        required = False,
                        default = 'car-prediction-220612')
    parser.add_argument('--data_folder',
                        type = str,
                        required = False,
                        default = 'output')
    parser.add_argument('--job-dir',
                        type=str,
                        required=False,
                        default='gs://car-prediction-220612/jobs')
    parser.add_argument('--file_name_train',
                        type = str,
                        default = 'train_data',
                        required = False)
    parser.add_argument('--file_name_test',
                        type = str,
                        required = False,
                        default = 'val_data')
    parser.add_argument('--max_train_steps',
                        type = int,
                        required = False,
                        default = 2000)

    ## TRAINING ARGUMENTS 
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3,
                        required=False)
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        required=False)
    
    ## HYPERPARAMETER ARGUMENTS
    parser.add_argument('--hidden_layers',
                        type=int,
                        default=3,
                        help='number of hidden layers in the mode',
                        required=False)
    parser.add_argument('--step_size',
                        type=float,
                        default=1e-4,
                        required=False)
    parser.add_argument('--activation_function',
                        type=str,
                        default='relu',
                        required=False)
    parser.add_argument('--layer_size',
                        type=int,
                        default = 5,
                        help =  """
                                Number of nodes in the first layer. 
                                The number of nodes will decrease over the layers.
                                We'll use a scale factor that will scale down the number nodes
                                in each layers
                                """,
                        required= False),
    parser.add_argument('--scale_factor',
                        type = float,
                        default = 0.7,
                        help = 'determines the number of nodes in a layer, by multiplying with the nodes in the last layer')
    parser.add_argument('--first_hidden_units',
                        type = int,
                        default = 2048,
                        help = 'Number of units in first layer'
    )

    # Prase the arge
    options = parser.parse_args()

    train_and_evaluate(
        data_dir =  options.data_folder,
        log_dir  =  options.job_dir,
        file_name_train = options.file_name_train,
        file_name_test = options.file_name_test,
        train_batch = options.batch_size,
        eval_batch = 1,
        max_train_steps = options.max_train_steps,
        arguments = options)
    
if __name__ == '__main__':
    main()


    