The repository for apnea diagnosis feature extraction on GCP cloudfunction

[Cloud Functions](https://console.cloud.google.com/functions/details/asia-east1/FeatureExtraction?hl=zh-tw&project=develop-335208)

## Step
1. Upload data to [GCS database](https://console.cloud.google.com/storage/browser/pranaq_database;tab=objects?forceOnBucketsSortingFiltering=false&hl=zh-tw&project=develop-335208&prefix=&forceOnObjectsSortingFiltering=false).
2. Generate ticket file (CSV).
3. Upload ticket to  [GCS tickets](https://console.cloud.google.com/storage/browser/feature_tickets;tab=objects?forceOnBucketsSortingFiltering=false&hl=zh-tw&project=develop-335208&prefix=&forceOnObjectsSortingFiltering=false).
4. Features will be made in 5 mins and store in [GCS features](https://console.cloud.google.com/storage/browser/pranaq_features;tab=objects?forceOnBucketsSortingFiltering=false&hl=zh-tw&project=develop-335208&prefix=&forceOnObjectsSortingFiltering=false).

## ticket file example



| patient type|
| -------- |
| name     |
| Lightoff Time |
| Start Record Time |
