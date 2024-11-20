# Bank Marketing Analysis

## Project Overview

The **Bank Marketing Analysis** project aims to predict whether a customer will subscribe to a term deposit based on various attributes from a bank's marketing campaign data. This classification problem uses machine learning techniques to analyze customer behavior and improve marketing strategies.

The dataset contains information about bank customers and the success of past marketing campaigns. The main objective is to predict if a customer will subscribe to a term deposit based on features such as age, job type, education, and various attributes related to the campaign.

## Dataset

The dataset consists of 17 columns before one-hot encoding. These columns represent different aspects of the customer and the marketing campaign:

### Bank Client Data:
- **age**: Age of the client (numeric).
- **job**: Job type (categorical: "admin", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services").
- **marital**: Marital status (categorical: "married", "divorced", "single").
- **education**: Education level (categorical: "unknown", "secondary", "primary", "tertiary").
- **default**: Credit in default (binary: "yes", "no").
- **balance**: Average yearly balance in euros (numeric).
- **housing**: Housing loan (binary: "yes", "no").
- **loan**: Personal loan (binary: "yes", "no").

### Last Contact of the Current Campaign:
- **contact**: Communication type used (categorical: "unknown", "telephone", "cellular").
- **day**: Last contact day of the month (numeric).
- **month**: Last contact month of the year (categorical: "jan", "feb", "mar", ..., "dec").
- **duration**: Duration of the last contact in seconds (numeric).

### Other Attributes:
- **campaign**: Number of contacts during the current campaign (numeric).
- **pdays**: Number of days since the last contact in a previous campaign (numeric; -1 means no prior contact).
- **previous**: Number of contacts before the current campaign (numeric).
- **poutcome**: Outcome of the previous campaign (categorical: "unknown", "other", "failure", "success").

### Target Variable:
- **y**: Whether the client subscribed to a term deposit (binary: "yes", "no").

## Key Features Affecting Campaign Success

- **Duration**: Longer contact durations correlate with a higher likelihood of subscription.
- **Age**: Older clients tend to be more likely to subscribe.
- **Balance**: Higher balance correlates with a greater chance of subscribing.
- **Job Type**: Certain job types like "management" and "entrepreneur" are associated with higher subscription likelihood.
- **Previous Outcome**: Clients with successful outcomes in previous campaigns are more likely to subscribe again.

## Key Features Affecting Campaign Success

Based on the analysis, several features play an important role in predicting whether a customer will subscribe to a term deposit. These include:

- **Duration**: The duration of the last contact is one of the most significant factors in predicting success. Longer contact durations generally lead to a higher likelihood of subscription.
- **Age**: Older clients tend to have a higher likelihood of subscribing to a term deposit, potentially due to financial stability.
- **Balance**: A higher balance in the client's bank account also correlates with a higher chance of subscribing to a term deposit.
- **Job Type**: Some job types, such as "management" and "entrepreneur", are associated with a higher likelihood of subscribing compared to others like "blue-collar" or "housemaid".
- **Previous Outcome**: If the client had a successful outcome in the previous campaign, they are more likely to subscribe again in the current campaign.

## Model Results

After training multiple machine learning models, including Logistic Regression and Random Forest, the model with the highest accuracy was found to be Random Forest. The evaluation metrics show the following:

- **Accuracy**: The model achieved an accuracy of approximately 85%, which is promising for predicting term deposit subscriptions.
- **Confusion Matrix**: The confusion matrix indicates that the model correctly identifies a significant number of both positive (subscribed) and negative (not subscribed) cases, though there is still room for improvement in reducing false positives and false negatives.
- **Precision and Recall**: Precision and recall values for the "yes" class (subscribed) were good, but the recall for the "no" class (not subscribed) could be improved.

## Technologies Used

This project uses the following technologies:

- **Python**: Programming language for data analysis and machine learning.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For implementing machine learning models and evaluation.
- **Matplotlib & Seaborn**: For data visualization.
- **Jupyter Notebook**: For interactive execution of the code and visualizing results.


## Installation Instructions

To run the project locally, follow these steps:

1. **Clone the repository to your local machine:**

    bash
    git clone https://github.com/saimqureshi656/Bank_Marketing_Analysis.git
    

2. **Navigate to the project directory:**

    bash
    cd Bank_Marketing_Analysis
    

3. **Set up a Python virtual environment (optional but recommended):**

    bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    

4. **Install the required dependencies:**

    bash
    pip install -r requirements.txt
    

    Alternatively, you can install the necessary libraries manually:

    bash
    pip install pandas numpy scikit-learn matplotlib seaborn
  
