# Genre-Predictor
Python package for predicting genres with a neural network created with Keras.

This package currently only predicts whether the given mp3 is rap or classical.

How to:
1. Delete placeholder.txt from all subfolders of train_files and eval_files - these files only exist to enable git commit of empty directories
2. Place whatever rap and classical training files (with extension .mp3 or .MP3) in train_files/Rap and train_files/Classical respectively
	- These files will be used to train the model
3. Place whatever rap and classical evaluation files (with extension .mp3 or .MP3) in eval_files/RapEval and eval_files/ClassicalEval respectively
	- These files will be used to assess model accuracy
4. Run Spectrogram.py
5. After Spectrogram.py finishes execution, run DeepNetModel.py
	- DeepNetModel.py will output the decimal accuracy (out of 1) of the model for each run and the average accuracy over all runs (# of runs is hard-coded as 1)
	- DeepNetModel.py will save each trained model to the models directory

