# Car Purchase Amount Prediction using Aritificial Neural Networks

## Project Overview

Imagine you're a car salesman trying to predict how much money a customer is willing to spend on a car. This project aims to develop a model that can make such predictions based on various customer attributes.

## Problem Statement

We want to create a predictive model that estimates the amount a customer will spend on a car. The prediction will be based on specific details about the customer, such as their age, annual salary, credit card debt, and net worth.

## Data Attributes

To make accurate predictions, we will use the following information about each customer:

- Customer Name
- Customer e-mail
- Country
- Gender
- Age
- Annual Salary
- Credit Card Debt
- Net Worth

## Target Prediction

The goal is to predict the **Car Purchase Amount** for each customer.

## Process Outline

### 1. Data Collection and Preparation

First, we collect data about customers, including their names, email addresses, countries, genders, ages, annual salaries, credit card debts, and net worths. This data is stored in a file called `Car_Purchasing_data.csv`.

### 2. Data Exploration

We start by exploring the data to understand it better:

- We look at the first and last few rows of the data to get an idea of what it looks like.
- We visualize the relationships between different attributes using a pair plot. This helps us see how various attributes relate to each other and to the car purchase amount.

### 3. Data Cleaning and Preparation

We clean the data by removing unnecessary columns like customer names and email addresses, which are not useful for prediction. We focus on the attributes that matter: age, annual salary, credit card debt, and net worth.

### 4. Data Normalization

To ensure all attributes contribute equally to the prediction, we normalize the data. Normalization scales the data to a standard range, making it easier for the model to learn from it.

### 5. Model Training

We split the data into training and testing sets. The training set is used to teach the model, while the testing set is used to evaluate its performance.

We use a neural network for the prediction. This network has layers that process the input data and learn the patterns associated with the car purchase amount. We train the model over several iterations (epochs) to improve its accuracy.

### 6. Model Evaluation

After training, we evaluate the model's performance:

- We plot the training and validation loss to see how well the model is learning.
- We make predictions on the testing set and compare them to the actual values to calculate the model's accuracy using a metric called the R2 score. The R2 score tells us how well our model's predictions match the actual values.

## Conclusion

This project demonstrates how we can use customer data to predict how much they might spend on a car. By carefully preparing the data and using advanced machine learning techniques, we have created a model that makes accurate predictions.

### Evaluation Result

Our model achieved an R2 score of **0.9892339**, indicating that it performs well in predicting car purchase amounts based on the provided customer attributes. This model can help car salesmen better understand their customers and tailor their sales strategies accordingly.
