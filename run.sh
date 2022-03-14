export TRAINING_DATA=input/train_folds.csv 
export TEST_DATA=input/test_cat2.csv
export SAMPLE_DATA=input/sample_submission.csv


export MODEL=$1

# Train each fold
# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train

# Test prediction
python -m src.predict
# sh run.sh 
# bash run.sh 
# ksh run.sh
# csh run.sh
# zsh run.sh 