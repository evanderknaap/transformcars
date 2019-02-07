from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
import tensorflow as tf

# Type of data that is coming 
# Tensorflow estimator require tf feature columns, which can be one of two things; 
# Numeric columns or categorical. Conventiently, in the case our data, it specifies the dtype
INPUT_HEADERS = ["Price","Year","Mileage","City","State","Vin","Make","Model"]
NUMERIC_FEATURE_KEYS = ["Year","Mileage","Price"]
CATEGORICAL_FEATURE_KEYS = ["City","State","Vin","Make","Model"]

# We'll read in the raw data from csv file. We'll need to decode this raw data
# to schema containing Tensors. We construct the schema here, which we'll pass
# to the decoder. Note that we'll generare need a second schema, contains
# features we add after reading the data. E.g., cross features
RAW_DATA_FEATURE_SPEC = dict(  
        [(name, tf.FixedLenFeature([],tf.float32)) for name in NUMERIC_FEATURE_KEYS] +
        [(name, tf.FixedLenFeature([],tf.string)) for name in CATEGORICAL_FEATURE_KEYS] 
    )
# Create dataset metadata from the feature spec
RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(RAW_DATA_FEATURE_SPEC))

# We'll create some additional helper lists, for the type of categorical
# numerical features 
LABEL_KEY = ["Price"]
DUMMY_FEATURES_KEYS = ["Vin"]
BUCKETIZED_FEATURES_KEYS = []
HASHED_FEATURE_KEYS = ["Model"]
CODING_FEATURE_KEYS = ['City','State','Vin','Make','Model']
 
# Create handy hyperparameters for the categorical features, to be put in buckets 
# This you will probably as parameters, later defined in the hyperparameter file
MILEAGE_BUCKETS =  [10000.0,25000.0,50000.0,75000.0] 
#YEAR_BUCKETS = [1990.0,1995.0,2000.0,2005.0,2010.0,2012.0,2014.0,2016.0,2017.0,2018.0]
BUCKETIZED_FEATURES = {}

