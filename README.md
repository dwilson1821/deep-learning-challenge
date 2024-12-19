# Deep Learning Challenge: Predicting Funding Success for Alphabet Soup

## Overview of the Analysis  

Alphabet Soup, a nonprofit foundation, wants a tool to help select applicants with the highest likelihood of success. Using machine learning, this analysis develops a binary classification model to predict whether an applicant will be successful if funded. By analyzing metadata from past applicants, we aim to provide a predictive framework to guide funding decisions effectively.

The analysis involves preprocessing the data, designing and training a neural network model, and optimizing the model to achieve a target accuracy of at least 75%.

---

## Repository Structure  

```  
deep-learning-challenge/  
│  
├── AlphabetSoupCharity.ipynb                # Preprocessing, training, and evaluation steps  
├── AlphabetSoupCharity_Optimization.ipynb   # Optimized model development  
├── Resources/                               # Dataset and HDF5 files  
│   ├── charity_data.csv                     # Alphabet Soup applicant data  
│   ├── AlphabetSoupCharity.h5               # Model file from the initial neural network  
│   ├── AlphabetSoupCharity_Optimization.h5  # Optimized model file  
├── README.md                                # Project documentation  
└── Images/                                  # Visuals for preprocessing and results (optional)  
```  

---

## Dataset  

The dataset contains historical funding information for over 34,000 organizations. Key columns include:  

- **Target Variable**:  
  - `IS_SUCCESSFUL`: Binary indicator of funding success (1 = success, 0 = failure).  

- **Feature Variables**:  
  - Metadata describing the organization, such as:
    - `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, etc.  

- **Excluded Variables**:  
  - `EIN` and `NAME`: Identification columns irrelevant for model training.  

---

## Analysis Steps  

### Step 1: Preprocessing the Data  

- Dropped the `EIN` and `NAME` columns.  
- Encoded categorical variables using `pd.get_dummies()`.  
- Combined rare categories into an "Other" label for columns with many unique values.  
- Split the data into features (`X`) and target (`y`).  
- Scaled the feature dataset using `StandardScaler`.  

### Step 2: Compile, Train, and Evaluate the Model  

- Designed a neural network with the following architecture:
  - Input layer: Based on the number of features.  
  - Hidden layers: Added one or more layers with activation functions to capture patterns.  
  - Output layer: Binary activation function for classification.  
- Compiled the model with `binary_crossentropy` loss and evaluated it on test data.  

### Step 3: Optimize the Model  

- Adjusted input data:
  - Modified bins for rare categories.
  - Dropped additional irrelevant columns.  
- Tuned model parameters:
  - Increased neurons in hidden layers.  
  - Added a second hidden layer.  
  - Experimented with different activation functions.  
  - Adjusted the number of epochs.  

---

## Results  

### Data Preprocessing  

- **Target Variable**:  
  - `IS_SUCCESSFUL`.  

- **Feature Variables**:  
  - `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`.  

- **Excluded Variables**:  
  - `EIN` and `NAME`.  

### Neural Network Model  

- **Architecture**:  
  - Hidden layers: X neurons (first layer), Y neurons (second layer, if added).  
  - Activation functions: ReLU for hidden layers, Sigmoid for output layer.  

- **Performance**:  
  - Initial model accuracy: X.XX%  
  - Optimized model accuracy: X.XX%  

### Steps to Increase Model Performance  

- Added more neurons to the first and second hidden layers.  
- Tuned the activation functions and experimented with Leaky ReLU.  
- Increased the number of epochs for additional training.  
- Adjusted preprocessing bins for categorical data.  

---

## Summary  

The final neural network model achieved an accuracy of **X.XX%** on the test data. While the model met [or did not meet] the target accuracy of 75%, several optimization strategies were employed, including feature adjustment and hyperparameter tuning.

### Recommendations  

If further optimization is required, consider:  

1. **Alternative Machine Learning Models**:  
   - Random Forests or Gradient Boosting models could capture non-linear relationships and interactions more effectively.  

2. **Feature Engineering**:  
   - Creating new features based on domain expertise could enhance predictive performance.  

3. **Deep Learning Enhancements**:  
   - Experiment with additional hidden layers or advanced techniques like dropout or batch normalization to improve model generalization.  

---

## How to Run the Project  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/your-username/deep-learning-challenge.git  
   cd deep-learning-challenge  
   ```  

2. **Set Up Environment**  
   Install required libraries:  
   ```bash  
   pip install pandas tensorflow scikit-learn matplotlib  
   ```  

3. **Run the Notebooks**  
   - Open `AlphabetSoupCharity.ipynb` and run all cells for the initial model.  
   - Open `AlphabetSoupCharity_Optimization.ipynb` for the optimized model.  

4. **View Model Results**  
   Evaluate the `.h5` files to see saved model configurations.  

---

## Tools and Technologies  

- **Python**: For data preprocessing and model development.  
- **TensorFlow/Keras**: Neural network modeling and optimization.  
- **Pandas & NumPy**: Data manipulation and analysis.  
- **scikit-learn**: Preprocessing and scaling.  
- **Google Colab**: Notebook environment for training and evaluation.  
