
## Data

- [x] EDA for ETL
- [x] create synth data generation script
- [x] create pipeline to generate and land new data in cloud storage
- [x] load initial data to cloud storage
- [x] create ETL pipeline for incoming training data
- [x] create ETL pipeline for incoming test/predict data 
- [ ] clear destinction for new data without inference column

## ML

- [x] EDA for ML
- [x] create pipline to load and preprocess data
    - [x] feature engineering
    - [x] data cleaning
    - [x] create funcs for loading pipelines
    - [x] create funcs for creating and saving pipelines with different steps
- [x] create MLOps structure
    - [x] set up experiment tracking
    - [x] set up metrics logging within modeltests
    - [ ] set up feature store (discarded)
    - [x] set up model registry

- [x] explore and tune models
    - [x] manual exploration
    - [x] automl
- [x] create pipeline for refitting and saving preprocessing Pipeline objects
- [ ] register good models
- [ ] create pipeline for training current best model
- [ ] create prediction pipeline
- [ ] create eval pipeline
- [ ] create pipeline for model re-evaluation/selection
- [ ] create system to monitor performance and trigger re-training and possibly re-selection of new models

## Infra


- [ ] set up precommit
- [x] set up MLOps components in databricks
- [x] put jobs into `.databricks/resources/`
- [ ] create github action CI/CD that deploys to dev/prod respectively
