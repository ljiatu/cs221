# cs221
CS221 Project

# How to run

- First ensure that all requirements are installed (PyTorch, nltk, pandas, etc.)
- Download the necessary data files from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
  - `test.csv`
  - `test_labels.csv`
  - `train.csv`
- In order to get the data into the format we used for the testing you need to run `preprocess.py` to a convenient location (this merges the `test.csv` and `test_labels.csv` files).
- In `main.py` ensure that the `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, and `TEST_OUTPUT_PATH` are set to the correct locations on your machine.
  - `TRAIN_DATA_PATH` should point to `train.csv`.
  - `TEST_DATA_PATH` should point to `processed.csv`.
- Next, you can run `main.py` to generate an output file to the `TEST_OUTPUT_PATH`
  - To test different models, simply replace the `model = ...` line with any of the other models (`CNN`, `KimCNN`, `LinearModel`, and `NeuralNet`).
  - You also have to use different extractors for different models.
    - For `LinearModel` and `NeuralNet`, you should use `Word2VecExtractor`.
    - For `CNN` and `KimCNN`, you should use `Word2Vec2DExtractor`.
- Finally, run `evaluate.py` to test the output file against the `test_labels.csv` to get final results printed in the console.
- Additionally, to check the window accuracy for our sliding window accuracy you can run the `run_window_accuracy.py` file.
  - Before doing this you should ensure that the saved model path set in `trainer.py` on line 79 `torch.save(...)` is set correctly for your machine. 
- After running this file you should see the overall accuracy printed to the console. 

# Folders and files

### datasets

- `kaggle_dataset.py`: loads all files for full comment text classification

- `kaggle_dataset_modified.py`: loads the test files for the window comment text classification, which is used in our toxicity source identification

### extractors

- `extractor.py`: abstract class defining the abstract extract method implemented used in the other extractors

- `window_accuracy.py`: reads in a saved model and then run our sliding window algorithm and perform an accuracy measurement

- `word2vec_2d_extractor.py`: extracts a 2D matrix out of each each sentence for CNN models

- `word2vec_extractor.py`: extracts a word2vec vector out of each sentence for non-CNN models

- `word_count_extractor.py`: extracts certain bad words from each sentence, used in the baseline model 

### models

- `cnn.py`: original CNN Model 

- `kim_cnn.py`: CNN Model that resembles the one designed by Y. Kim

- `linear_model.py`: simple linear model

- `neural_net.py`: simple neural net model used with 2-4 layers

### utils

- `evaluator.py`: evaluates a model output file against the results using the ROC AUC score 

- `label.py`: represents the label given to a certain comment

- `trainer.py`: runs the process of training, testing, evaluating, and writing results for a given model

### general

- `count_length.py`: counts sentence length, which was used to determine matrix dimension for CNN models

- `error_picker.py`: finds examples of incorrect classifications in our results

- `evaluate.py`: evaluates two files using the evaluator.py file in utils

- `main.py`: runs through the basic process of loading, training, and testing the data

- `preprocess.py`: preprocesses test data and combines both comment text and labels into one file. Also filters out rows that are not used for testing 

- `requirements.txt`: outlines the requirements for running the project

- `run_window_accuracy.py`: checks the accuracy of the sliding window algorithm
