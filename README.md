# CLUZH submission for VarDial 2017 Workshop Task "German Dialect Identification" (GDI)

STILL WORK IN PROGRESS

This repository contains the code and data for reproducing our approach as described in our paper.

### Prerequisites for running the code under Linux/MacOS:
  - our tested environment: Anaconda 4.3.0 with Python 3 https://www.continuum.io/downloads and Keras 1
  - wapiti command line tool https://wapiti.limsi.fr
  - make 


### Steps to reproduce our results
Execute all commands from the top directory of GDI-task-2017 directory.
   - ``python3 lib/generate_cross_validation_data.py`` 
     - populates the directory ``./cv.d`` with data splits
   	 - creates the first global split into ``test.tsv`` and ``train.tsv`` used for the LSTM experiments (no cross-validation there) 
   	 - creates the cross-validation data splits using ``train.tsv`` as input: 
   	 	test_N.tsv and train_N.tsv (0 <= N <= 9) 
   	 
   - ``make -f run2.mk target``
     - train and evaluate all folds of CRF run 2 using wapiti
     
   - ``python3 lib/dataprep_runs1and3.py``
     - populates the directory ``preprocessed_data.d``
     - create all the models and output for runs 1 (Naive Bayes) and 3 (NB, SVM, CRF ensemble) (using the results of run 2)
     - creates the data variants (augmented data, replacements) for the LSTM models
     
   - ``for var in model+charrep+augm model+charrep-augm model-charrep+augm model-charrep-augm ; do python3 lib/LSTM_models.py $var ; done``
	 - trains 4 different LSTM models *with* character embeddings as described in the paper
	 - takes some hours on non-GPUs
 
   - ``for var in model+charrep+augm model+charrep-augm model-charrep+augm model-charrep-augm ; do  python3 lib/LSTM_models_emb.py $var ; done``
	 - trains 4 different LSTM models *without* character embeddings as described in the paper
	 - takes some hours on non-GPUs

