
#### dag 1
# pull from dvc
# load model 
## if no model - train from scratch - entire dataset
## train model
## evaluate model
## save the model

## if model - fine tune, get date until which it was trained
## split dataset
## train model only for dataset split based on dates
## evaluate model
## save the model

# manual trigger
# check evaluation metrics
## if less than threshold -> hyperparamater fine tuning
## retraining
## evaluatiion
## save
####

#### dag 2
# pull trained model
# check data bias in this trained model
# apply data bias mitigation
# train
# evaluate 
# save
####