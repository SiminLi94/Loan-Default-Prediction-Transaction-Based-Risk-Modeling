# Loan Default Prediction: Transaction-Based Risk Modeling

## Project Overview

The goal of this project is to build a machine learning model that predicts the probability that a given customer will default on a loan. The model uses customer transaction history to make predictions, where each transaction is classified as either a debit (money going out of the account) or a credit (money coming into the account). This prediction aims to help financial institutions assess loan risk before granting loans.

## Customer Profile (Attributes)

The customer profile consists of the following attributes for each customer:

- **Id**: A unique identifier for each customer.
- **dates**: Dates of each transaction.
- **transaction_amount**: A numpy array of credits and debits. The length varies across customers, and this array contains the primary information used for predictions.
- **days_before_request**: The number of days before the loan request corresponding to each transaction.
- **loan_amount**: The amount loaned to the customer by the bank.
- **loan_date**: The date the loan was given.

### Outcome:
- **isDefault**: Indicates whether the customer paid back the loan (isDefault = 0) or did not pay back the loan (isDefault = 1). This information is available for the first 10,000 customers. The goal of this project is to predict the probability of `isDefault` for the remaining 5,000 customers.

### Dataset:
- **Training Data**: Instances 0 - 9999.
- **Test Data**: Instances 10000 - 14999 (no `isDefault` column in the test set).

The dataset can be accessed via the following link:  
[Download Dataset](https://drive.google.com/file/d/1oPSNCYeCVGJsTX60X-PW088R8S0AMmeT/view?usp=sharing)
