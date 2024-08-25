# Telemarketing Campaign Optimization for Electric Bank

## **Overview**

This project focuses on developing a machine learning model to predict whether a customer will subscribe to a term deposit after being contacted through a telemarketing campaign. The goal is to improve the effectiveness of Electric Bank's telemarketing strategy by increasing the conversion rate and enhancing the return on marketing investment (ROMI) to match or exceed industry benchmarks.

## **Context**

Electric Bank, a prominent financial institution, has been conducting telemarketing campaigns to encourage customers to subscribe to fixed-term deposits. Despite these efforts, the bank's current conversion rate of **11.27%** lags behind the top performers in the financial industry, who are converting at a rate of 23% (Ruler Analytics). This gap highlights the need for Electric Bank to improve its telemarketing strategy to reach the success levels seen by industry leaders.

Electric Bank’s ROMI of **144.92%** indicates that for every euro spent on telemarketing, the bank generates an additional 1.45 euros in profit. However, this figure is below the industry benchmark of 5:1 or 500%, which is considered a good ROMI (Improvado). This suggests that there is substantial room for improvement in the profitability of the bank's marketing efforts, as achieving a higher ROMI is essential for ensuring that marketing investments yield substantial returns.

## **Problem Statement**
The main challenge is to refine Electric Bank's telemarketing approach to increase the conversion rate and ROMI. The current conversion rate of **11.27%** and a ROMI of **144.92%** indicate potential, but there is significant room for improvement. The objective is to develop a machine learning model that accurately predicts which customers are likely to subscribe to a term deposit, allowing the bank to focus its efforts on high-potential leads, with the ultimate goal of achieving conversion rates similar to top performers and maximizing ROMI.

## **Goal**

The primary objectives of this project are:
- **Achieve Top Performer Conversion Rates**: Improve the precision of targeting potential subscribers to increase the conversion rate to match the **top performers in the industry at 23%**.
- **Maximize ROMI**: Enhance the return on marketing investment by ensuring that the profit generated from successful term deposit subscriptions significantly exceeds the costs of the telemarketing campaigns, aiming for a ROMI closer to the **industry benchmark of 500%**.

## Evaluation Metrics
- **Precision**: Chosen because the cost of false positives (predicting that a client will subscribe when they actually won’t) is high. With precision, we can focus on ensuring that the clients we predict as likely to subscribe are indeed the ones who will do so, which directly supports the goal of reducing unnecessary telemarketing efforts.

## Cost Analysis
Below are the net gains associated with each possible prediction outcome:

- **True Positive (TP):**
  - Net Gain: 500 euros (deposit revenue) − 23 euros (call cost) = 477 euros

- **False Positive (FP):**
  - Net Gain:  0 euros (no revenue) − 23 euros (call cost) = − 23 euros

- **True Negative (TN):**
  - Net Gain: 0 euros (no revenue, no cost)

- **False Negative (FN):**
  - Net Gain: 0 euros (no revenue, no cost)

## Stakeholder

The primary stakeholder for this project is the **Marketing Team Electric Bank**.

## Data Understanding

* For this project, we use a dataset that describing Portugal bank marketing campaigns conducted using telemarketing, offering customers to place a term deposit. If after all marking afforts customer had agreed to place deposit - target variable marked 'yes', otherwise 'no'.  

* Each row represents information from a customer and the socio-economic circumstances of the previous marketing campaign.

(source : https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)

### **Attribute Information**


**Customer Demographic**
<br>

| Attribute | Data Type | Description |
| --- | --- | --- |
|Age |Integer | age of customer |
|Job |Text | type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown") |
|Marital |Text | marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed) |
|Education |Text | level of education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown") |
|Default |Text | has credit in default? (categorical: "no","yes","unknown") |
|Housing |Text | has housing loan? (categorical: "no","yes","unknown") |
|Loan |Text | has personal loan? (categorical: "no","yes","unknown") |
<br>

**Information During This Campaign**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Contact |Text | contact communication type (categorical: "cellular","telephone") |
|Month |Text | last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec") |
|Day_of_week |Text | last contact day of the week (categorical: "mon","tue","wed","thu","fri") |
|Duration |Integer | last contact duration, in seconds. Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known |
|Campaign |Integer | number of contacts performed during this campaign and for this customer (numeric, includes last contact) |
<br>

**Information From Previous Campaign**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Pdays |Integer | number of days that passed by after the customer was last contacted from a previous campaign (numeric; 999 means customer was not previously contacted) |
|Previous |Integer | number of contacts performed before this campaign and for this customer |
|Poutcome |Text | outcome of the previous marketing campaign (categorical: "failure","nonexistent","success") |
<br>

**Customer Socio-Economic**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Emp.var.rate |Float | employment variation rate - quarterly indicator |
|Cons.price.idx |Float | consumer price index - monthly indicator |
|Cons.conf.idx |Float | consumer confidence index - monthly indicator |
|Euribor3m |Float | euribor 3 month rate - daily indicator |
|Nr.employed |Float | number of employees - quarterly indicator |
<br>

**Target**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Y |Text | has the customer subscribed a term deposit? (binary: "yes","no") |

## Data Preparation

**Data Cleaning**
* Removed duplicate records.
* Deleted features with mostly missing values (e.g., the 999 value in the pdays column).
* Handled outliers.
* Checking label ambiguities.
* Checking cardinality
* Regrouped certain features for better analysis.
* Applied binning with KBinsDiscretizer using the quintile strategy.

**Feature Engineering**
* Extracted features such as Loan Status from Housing_Loan and Personal_loan
* Conducted feature selection using Power Predictive Score (PPS).
* Applied feature transformation, including custom encoding, binary encoding, one-hot encoding, label encoding, and scaling using Robust Scaler.

## Model Choosen: XGBoost Classifier
After evaluating the models' performance, XGBoost Classifier was chosen over Gradient Boosting and any others model.

## Conclusion
We developed the best model using data cleaning, feature extraction, preprocessing techniques, and model benchmarking which is the XGBoost Classifier. The model achieves a high accuracy of 90% and a precision of 84%, meaning it is effective at correctly predicting deposits, and from the model evaluation using the Learning Curve and Brier Score, the model shows the best performance and is expected to give accurate predictions. In summary, **the model can help Electric Bank improve its Conversion Rate 7.4 times and Return On Marketing Investment (ROMI) 11.84 times from original state by effective targeted telemarketing.**

## Separate Model for New Customer Acquisition

We also developed a separate model for predicting new customer acquisitions, particularly for cases where customer data is newly available and has not been subjected to any previous campaign. The **Naive Bayes** model, with **SMOTE oversampling**, scored **0.845** and shows potential for increasing conversion rates by **6.61 times** and reducing telemarketing costs by **7.82 times**.