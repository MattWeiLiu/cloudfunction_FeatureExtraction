The repository for apnea diagnosis feature extraction on GCP cloudfunction

[Cloud Functions](https://console.cloud.google.com/functions/details/asia-east1/FeatureExtraction?hl=zh-tw&project=develop-335208)

## Steps
1. Upload data to [GCS database].
2. Generate ticket file (CSV).
3. Upload ticket to  [GCS tickets].
4. Features will be made in 5 mins and store in [GCS features].

## ticket file example



| patient type|
| -------- |
| name     |
| Lightoff Time |
| Start Record Time |
