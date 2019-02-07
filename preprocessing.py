"""
Pipeline that reads csv data on car predictions data from a local folder or buck on GCP,
and writes the transformed data in TFRecords. TransformOutput is also written out tot disk, 
so that the preprocessing computations can be resuesed for preprocessing, training and serving. 
Runs locally and on Google Cloud.

To run locally
python preprocessing.py \
--output_folder output \
--file_name_train train_data \
--file_name_test val_data \
--save_main_session True \
--runner DirectRunner

To run on DataFlow 
python preprocessing.py \
--project car-prediction-220612 \
--temp_location gs://car-prediction-220612/temp \
--staging_location gs://car-prediction-220612/staging \
--input_folder gs://car-prediction-220612/source \
--output_folder gs://car-prediction-220612/output \
--file_name_train train_data \
--file_name_test val_data \
--region europe-west1  \
--num_workers 5 \
--runner DataFlowRunner \
--save_main_session True \
--requirements_file requirements_dataflow.txt \
--job_name transform

By: Eric van der Knaap
"""

from __future__ import absolute_import
from __future__ import division

import apache_beam as beam 
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import GoogleCloudOptions 
import argparse

import tempfile
import os
import time

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

# Import metadata ion the features
from input_metadata import INPUT_HEADERS
from input_metadata import CATEGORICAL_FEATURE_KEYS
from input_metadata import BUCKETIZED_FEATURES_KEYS
from input_metadata import CODING_FEATURE_KEYS
from input_metadata import HASHED_FEATURE_KEYS
from input_metadata import BUCKETIZED_FEATURES
from input_metadata import LABEL_KEY
from input_metadata import RAW_DATA_FEATURE_SPEC
from input_metadata import RAW_DATA_METADATA

class PreprocessOptions(GoogleCloudOptions):

    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument('--input_folder',
        					type=str,
                            help='Folder that contrain the raw data',
                            required=False,
                            default=''),
        parser.add_argument('--output_folder',
        					type=str,
                            help='Output folder that contains transfered data TransformOutput',
                            required=False,
                            default='output'),
        parser.add_argument('--file_name_train',
                            type=str,
                            help='Filename of the input train data, csv format is expected',
                            required=True,
                            default='train_data'),
        parser.add_argument('--file_name_test',
                            type=str,
                            help='Filename of the input test data, csv format is expected',
                            required=True,
                            default='test_data'),
        parser.add_argument('--runner',
                            type=str,
                            help='Runner used for pipeline execution',
                            required=True,
                            default='DirectRunner')
        
class CreateKVPair(beam.DoFn):
    """Creates key value pair of the input element where the key equals the value"""
    def process(self, element):
        try:
            kv = [(element,element)]
        except:
            kv = []
        return kv   

class FilterDuplicates(beam.DoFn):
    """Grabs and returns the first key"""
    """
        Args:   list of key value pairs
        Output: list of only the first key    
    """
    def process(self, element):
        try:
            # Grab the dic(s), if there are multiple and throw out the double one
            yield element[0]
        except:
            yield []
        return

class CleanDuplicates(beam.PTransform):
    def expand(self, pcol):
        return (pcol
            # Create KV pair where key equals value of the input string
            |'Create KV of input string'   >> beam.ParDo(CreateKVPair())
            # Group by element, to find identical strings
            |'GroupBy by string'            >> beam.GroupByKey()
            # Filterout the duplicate input lines
            |'Filter Duplicates'            >> beam.ParDo(FilterDuplicates())
            ) 

class PrintCollection(beam.DoFn):
    def process(self, element):
        try:
            print element
        except:
            print ""
        return

def preprocessing_fn(inputs):
    """Preprocess the input features"""

    # Remove all feature keys for debugging 
    outputs=inputs.copy()

    # Normalize the numeric features using the Z-score 
    outputs[LABEL_KEY[0]] = tft.scale_to_z_score(outputs["Price"])
    outputs['Mileage'] = tft.scale_to_z_score(inputs['Mileage'])
    outputs['Year'] = tft.scale_to_z_score(inputs['Year'])
      
    # Compute the vocabulary of the categorical features
    for key in CODING_FEATURE_KEYS:
        tft.vocabulary(inputs[key], vocab_filename=key)

    # # CATEGORICAL FEATURES
    # # Creat bucketized features for year and mileage
    # for key in BUCKETIZED_FEATURES:
    #     boundaries =  BUCKETIZED_FEATURES[key] # List of boundaries
    #     outputs[key] = tft.apply_buckets(
    #         inputs[key],
    #         tf.reshape(tf.convert_to_tensor(boundaries),[len(boundaries),1]),
    #         name=key)       
    
    # # TO DO: create hashed feature for the spare categorical features
    # # Maybe embeddings for Model & Make. E.g.,
    # for key in HASHED_FEATURE_KEYS:
    #     outputs[key] = tft.hash_strings(inputs[key], hash_buckets=100)

    return outputs
    
def transform_data(input_dir,output_dir,temp_dir,file_name_train,file_name_test, options):
    """ Transforms raw data in csv format to transformed dataset in TFRecords
        Args:
            working_dir: directory to story transformed data and TransformOutput 
            temp_dir: path to temp file on disk or GCS
            data_dir: directory where the raw data is stored   
    """

    with beam.Pipeline(options = options) as p:
        with tft_beam.Context(temp_dir):

            # Create a decoder, that turns string of CSV data, into a dictionary of Tensors
            # The schema describes how to convert data, but not the order in which they appear in the raw data 
            # We pass in INPUT_HEADERS for that
            converter = tft.coders.CsvCoder(INPUT_HEADERS, RAW_DATA_METADATA.schema)

            # Read in raw data, clean it, decode to a list of dictionaries with tensor values, to raw_features_spec
            raw_data = (p
                    |'ReadCSV'                  >> beam.io.ReadFromText(os.path.join(input_dir,file_name_train+'.csv'), skip_header_lines=True)
                    |'Remove duplicate lines'   >> CleanDuplicates()
                    |'Decode'                   >> beam.Map(converter.decode)
                    )

            # put the raw data in tuple with the RAW_DATA_SCHEMA metdata
            raw_dataset = (raw_data,RAW_DATA_METADATA) 
            
            # print for depugging
            #_= raw_data | 'Print raw data' >> beam.ParDo(PrintCollection())
                        
            # Create a transform_fn. This function holds the logic to preprocess data for training, and serving
            # and can be reused to transform train and test set 
            transform_fn = (raw_dataset |'Create transform function' >> tft_beam.AnalyzeDataset(preprocessing_fn))
            
            # Transform dataset, including meta_data -> Note that a new schema is created 
            # through the Categorical columns
            transformed_dataset = ((raw_dataset,transform_fn) | 'Transform train_dataset' >> tft_beam.TransformDataset())         
            
            # Extract the transformed data and new metadata from the dataset 
            transformed_data, transformed_metadata = transformed_dataset

            # Print transformed dataset
            #_ = transformed_data |'Print transformed' >> beam.ParDo(PrintCollection())

            # Create a coder with the transformed schema, to write to disk in TFRecord format
            transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)

            _  = (transformed_data  
                |'Encode Train Data' >> beam.Map(transformed_data_coder.encode)
                |'Write train data' >> beam.io.WriteToTFRecord(os.path.join(output_dir,file_name_train)))
        
            # TO DO : Transform the Test Data
            # Write the output function to disk
            _ = (transform_fn |'Write transform_fn to disk' >> tft_beam.WriteTransformFn(output_dir))

            # Read in Raw test data
            raw_test_data = (p 
                            | 'Read test_data'         >> beam.io.ReadFromText(os.path.join(input_dir,file_name_test+'.csv'), skip_header_lines=True)
                            | 'Remove test duplicates' >> CleanDuplicates()
                            | 'Decode test_data'       >> beam.Map(converter.decode)
                            )

            # Now that we have read in our data, we can use our transform_fn to to prerocess
            # We pass the transformed function together with the raw data and get back the processed test data
            raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)
            transformed_test_dataset = ((raw_test_dataset, transform_fn) | 'Transform test_dataset' >> tft_beam.TransformDataset())
            transformed_test_data, _ = transformed_test_dataset

            # We use the encoder, we built from our train data, to encode our test data and write to disk
            # in TFRecord format
            _ = (transformed_test_data 
                    | 'Encode transform test data'  >> beam.Map(transformed_data_coder.encode)
                    | 'Write test data to disk'     >> beam.io.WriteToTFRecord(os.path.join(output_dir, file_name_test))
            )

    return

def main():
    
    # Subclass of GoogleCloudOptions. Added folder and filenames for 
    # input and output of the pipeline
    options = PreprocessOptions()

    if options.runner == 'DataFlowRunner':
        transform_data(
            input_dir  = options.input_folder,
            output_dir = options.output_folder, 
            temp_dir   = options.temp_location, 
            file_name_train = options.file_name_train, 
            file_name_test  = options.file_name_test,
            options = options) 
    else:  
        transform_data(
            input_dir   = options.input_folder,
            output_dir  = options.output_folder,
            temp_dir    = tempfile.mkdtemp(),
            file_name_train = options.file_name_train,
            file_name_test = options.file_name_test,
            options = options)

if __name__ == '__main__':
    main()


    