
## Data

- [ ] EDA for ETL
- [ ] create synth data generation script
- [ ] create pipeline to generate and land new data in cloud storage
- [x] load initial data to cloud storage
- [ ] create ETL pipeline for incoming training data
- [ ] create ETL pipeline for incoming test/predict data 

## ML

- [ ] EDA for ML
- [ ] create pipline to load a preprocess data
    - [ ] feature engineering
    - [ ] data cleaning
- [ ] create MLOps structure
    - [ ] set up experiment tracking
    - [ ] set up feature store
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
- [ ] put jobs into `.databricks/resources/`
- [ ] create github action CI/CD that deploys to dev/prod respectively
