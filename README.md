# cs221
CS221 Project

# How to run

- First ensure that all requirements are installed (PyTorch, nltk, pandas, etc.)
- Download the necessary data files from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data 
  - `test.csv`
  - `test_labels.csv`
  - `train.csv`
- In order to get the data into the format we used for the testing you need to run `preprocess.py` to a convenient location (this merges the test and test_labels files)
- In `main.py` ensure that the `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, and `TEST_OUTPUT_PATH` are set to the correct locations on your machine.
  - `TRAIN_DATA_PATH` should point to `train.csv`
  - `TEST_DATA_PATH` should point to `processed.csv`
- Next, you can run `main.py` to generate an output file to the `TEST_OUTPUT_PATH`
  - To test different models simply replace the `model = ...` line with any of the other models (`CNN`, `KIMCNN`, `LinearModel`, and `NeuralNet`)
- Finally, run `evaluate.py` to test the output file against the `test_labels.csv` to get final results printed in the console.
- Additionally, to check the window accuracy for our sliding window accuracy you can run the `run_window_accuracy.py` file.
  - Before doing this you should ensure that the saved model path set in `trainer.py` on line 79 `torch.save(...)` is set correctly for your machine. 
- After running this file you should see the overall accuracy printed to the console. 

# Folders and files

### datasets

- kaggle_dataset.py: Dataset subclass used to load all files for full comment text classification

- kaggle_dataset_modified.py: Dataset subclass use to load the test files for the window comment text classification used in our toxicity source identification

### extractors

- extractor.py: abstract class defining the abstract extract method implemented used in the other extractors

- window_accuracy.py: class to read in a saved model and then run our sliding window algorithm and perform an accuracy measurement

- word2vec_2d_extractor.py: extractor used to extract a 2D matrix out of each each sentence

- word2vec_extractor.py: extractor used to extract a word2vec vector out of each sentence

- word_count_extractor.py: extractor used in our baseline model that extractors certain bad words from each sentence 

### models

- cnn.py: original CNN Module 

- kim_cnn.py: Kim CNN Module designed for text classification

- linear_model.py: simple linear model Module

- neural_net.py: simple neural net Module used with 2-4 layers

### outputs

- All files relate to the output predictions from a given date

### utils

- evaluator.py: used to evaluate an model output file against the results using the roc_auc score 

- label.py: used to represent the label given to a certain comment

- trainer.py: runs the process of training, testing, evaluating, and writing results from a loaded file for a given model

### general

- count_length.py: test file used for counting lines of data

- error_picker.py: used to find examples of incorrect classifications in our results

- evaluate.py: evaluates two files using the evaluator.py file in utils

- main.py: runs through the basic process of loading, training, and testing the data

- preprocess.py: used to preprocess data into a format suitable for loading 

- requirements.txt: outlines the requirements for running the project

- run_window_accuracy.py: checks the accuracy of the sliding window algorithm
