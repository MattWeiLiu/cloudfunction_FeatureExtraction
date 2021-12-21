# Copyright 2018, Google, LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# [START functions_helloworld_storage_generic]

from google.cloud import storage
import numpy as np
import sys
import warnings

from Preprocess import DataPreprocess
from FeatureExtraction import FeatureExtraction
warnings.filterwarnings("ignore")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def blob_metadata(bucket_name, blob_name):
    """Get a blob's metadata."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    try:
        contenido = blob.download_as_string()
        s = str(contenido,'utf-8')
        return s.split()
    except:
        return None

def GCS_trigger_me(event, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.
    Args:
        event (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """
    try:
        bucket_name = event['bucket']
        blob_name = event['name']
        patient_name = blob_name.split('.')[0]

        destination_bucket_name = 'pranaq_features'
        source_file_name = '/tmp/test.npz'
        destination_blob_name = '{}.npz'.format(patient_name)
        
        P = DataPreprocess()
        F = FeatureExtraction()
        
        patient_type, name, lightoff_time, startRecord_time = blob_metadata(bucket_name, blob_name)
        lightoff_time, startRecord_time = float(lightoff_time), float(startRecord_time)
        state, Channels = P.trigger_by_cloudfunction(patient_type, name, lightoff_time, startRecord_time)
        print ('Preprocess Complete')
        AR_THO, AR_ABD, FR_THO, FR_ABD, SDM_T_THO, SDM_T_ABD, SDM_P_THO, SDM_P_ABD,mu_SpO2, drop_SpO2, min_SpO2, max_SpO2 = F.CF_FeatureExtraction(state, Channels)
        print ('Feature Extraction Complete')
        np.savez(source_file_name, 
            AR_THO=AR_THO, AR_ABD=AR_ABD, FR_THO=FR_THO, 
            FR_ABD=FR_ABD, SDM_T_THO=SDM_T_THO, SDM_T_ABD=SDM_T_ABD, 
            SDM_P_THO=SDM_P_THO, SDM_P_ABD=SDM_P_ABD, mu_SpO2=mu_SpO2, 
            drop_SpO2=drop_SpO2, min_SpO2=min_SpO2, max_SpO2=max_SpO2)
        upload_blob(destination_bucket_name, source_file_name, destination_blob_name)
    except Exception as err:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        line_number = exception_traceback.tb_lineno
        print("Exception type: ", exception_type)
        print("Error: ", err)
        print("Line number: ", line_number)
        print ("Event: ", event)