# CONITACS-EEGNetModel


# README

## EEG Classification using EEGNet

This project implements a deep learning approach for classifying EEG signals using the EEGNet architecture. The code processes EEG data, trains a convolutional neural network (CNN) on it, and evaluates the model's performance.

### Requirements

To run this code, you need to have the following libraries installed:

- mne
- pandas
- tqdm
- matplotlib
- numpy
- seaborn
- scikit-learn
- keras
- tensorflow

You can install these libraries using pip:

```bash
pip install PyWavelets numpy==1.24.2 tqdm keras==2.15.0 tensorflow==2.15.0 scikit-learn==1.2.2 mne==1.4.2
```

### Project Structure

- `read_data(file_path)`: Reads and preprocesses EEG data from a GDF file.
- `preprocessing(X_tr, labels)`: Scales the data and splits it into training and testing sets.
- `EEGNet(nb_classes, Chans, Samples, regRate, dropoutRate, kernLength, numFilters, dropoutType)`: Defines the EEGNet model architecture.
- `train_model(model, x_train, y_train, x_test, y_test)`: Compiles and trains the model.
- `k_fold_split(X_tr, labels, n_splits)`: Splits the data into k folds for cross-validation.
- `evaluate_model(model, x_test, y_test)`: Evaluates the model's performance and displays a confusion matrix and classification report.
- `main()`: Main function to read data, preprocess it, train the model, and save the trained model.

### Usage

1. **Prepare your data**: Ensure you have your EEG data files in GDF format. Modify the `file_paths` list in the `main` function to include the paths to your data files.

2. **Run the main script**: Execute the script to start the training and evaluation process.

```bash
python your_script_name.py
```

3. **Model Training and Evaluation**: The script will read the EEG data, preprocess it, train the EEGNet model using k-fold cross-validation, and evaluate the model's performance. The final trained model will be saved to disk.

### Saving the Model

The trained model architecture and weights will be saved as:

- `EEGNET_four_classes.json`
- `EEGNET_four_classes_v1.h5`

### Example

Here is a brief overview of how the code works:

1. **Read and preprocess the data**: The `read_data` function reads the EEG data from GDF files, applies filtering, and extracts epochs and labels. The `preprocessing` function scales the data and splits it into training and testing sets.

2. **Define the EEGNet model**: The `EEGNet` function defines the CNN architecture for classifying EEG data.

3. **Train the model**: The `train_model` function compiles and trains the model on the preprocessed data.

4. **Evaluate the model**: The `evaluate_model` function evaluates the model's performance using a confusion matrix and classification report.

5. **Main function**: The `main` function coordinates the entire process, including reading data, preprocessing, training, and saving the model.

### Notes

- Ensure that the EEG data files are correctly formatted and the paths are accurately specified.
- Modify the parameters in the `EEGNet` function to tune the model according to your needs.
- The script uses k-fold cross-validation to ensure robust model training and evaluation.

### Acknowledgements

This project utilizes the EEGNet architecture and various machine learning libraries to achieve accurate classification of EEG signals. Special thanks to the developers of these libraries and the researchers who contributed to the EEGNet model.

---

Feel free to modify this README to better suit your project's needs. If you encounter any issues or have questions, please refer to the respective library documentation or seek assistance from the community.
