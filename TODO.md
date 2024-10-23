
## Data

- [x] EDA for ETL
- [x] create synth data generation script
- [ ] create pipeline to generate and land new data in cloud storage
- [x] load initial data to cloud storage
- [x] create ETL pipeline for incoming training data
- [x] create ETL pipeline for incoming test/predict data 

## ML

- [x] EDA for ML
- [x] create pipline to load and preprocess data
    - [x] feature engineering
    - [x] data cleaning
    - [x] create funcs for loading pipelines
    - [x] create funcs for creating and saving pipelines with different steps
- [ ] create MLOps structure
    - [ ] set up experiment tracking
    - [ ] set up feature store (discarded)
    - [ ] set up model registry

- [ ] explore and tune models
    - [ ] manual exploration
    - [x] automl
- [ ] register good models
- [ ] create pipeline for training current best model
- [ ] create prediction pipeline
- [ ] create eval pipeline
- [ ] create pipeline for model re-evaluation/selection
- [ ] create system to monitor performance and trigger re-training and possibly re-selection of new models

## Infra


- [ ] set up precommit
- [ ] set up MLOps components in databricks
- [x] put jobs into `.databricks/resources/`
- [ ] create github action CI/CD that deploys to dev/prod respectively
