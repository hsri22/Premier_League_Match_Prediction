# Premier League Match Prediction using RandomForest Classifier

This repository contains Python code for analyzing Premier League match data using a RandomForest Classifier. The code leverages pandas, matplotlib, numpy, seaborn, and scikit-learn libraries to process and analyze match data from the Premier League.

**Key Features:**
* **Data Import and Preprocessing:** The code reads match data from a CSV file and performs preprocessing steps such as dropping irrelevant columns, adding new columns (e.g., 'Winner', 'gd', etc.), and converting columns to appropriate data types.

* **Data Transformation and Feature Engineering:** The code creates numerical codes for teams, calculates rolling averages for specific features, and maps match results to numerical values. It also removes rows with 'Draw' results for simplicity.

* **Model Training and Prediction:** A RandomForest Classifier model is created and trained using the training data. The trained model is used to predict match results for the test data.

* **Accuracy and Metrics Calculation:** The code calculates accuracy and other metrics such as overall accuracy, win accuracy, loss accuracy, precision for win and loss outcomes, and more.

* **Visualization of Results:** The code visualizes team-specific metrics such as wins and losses using bar plots.

* **Team Analysis and Prediction Ranking:** The code ranks teams based on their predicted win ratios.



**Usage:**
1. Install the required libraries by running: pip install pandas matplotlib numpy scikit-learn seaborn 
2. Modify the file path to point to your Premier League match data CSV file.
3. Run the code to execute each section sequentially.
4. Examine the printed results, accuracy metrics, visualizations, and team ranking.

**Note:**

This code provides an end-to-end solution for analyzing Premier League match data and predicting match outcomes using a RandomForest Classifier. Customize file paths and other parameters according to your dataset and analysis requirements.

For a detailed understanding of the code and its implementation, refer to the provided comments and consider further optimizations based on your specific dataset and use case.

Feel free to explore, modify, and adapt this code to your specific sports analytics projects.
