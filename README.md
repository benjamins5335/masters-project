
# How to Run

1. **Navigate to the CNN Directory:**
   - Once the `masters-project` folder has been downloaded, open your terminal and navigate to the `cnn` folder by running the following command:
     ```bash
     cd cnn
     ```

2. **Download the Dataset:**
   - Download the dataset from [Zenodo](https://zenodo.org/records/10513773) into the `cnn` directory.
   - Alternatively, you can generate all images and preprocess them to form the dataset by running the following command:
     ```bash
     bash generate_all_images.sh
     ```
   - **Note:** Generating the dataset will take approximately 16 days with a standard GPU.

3. **Train the Model (Optional):**
   - After the dataset is ready, you can train the model using the following command:
     ```bash
     bash train.sh
     ```
   - The model’s hyperparameters can be modified in the `config.json` file before starting the training process.
   - Once training is complete, the model will be saved to the `models` directory.

4. **Model Evaluation:**

   - You can evaluate the model on the test set by running the following command:
     ```bash
     bash evaluate.sh <PATH TO MODEL>
     ```
   - A pre-trained model named `model.pth` already exists in the `models` directory.
