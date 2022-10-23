# Build-Ideal-Model-For-Predicting-Loan-Status-Customers
In this project, I try to Build most ideal model for predicting Loan Status Customers in dataset by Performing Supervised Classification model Machine Learning.

## Overview
**Introduction**

Hi, I'm Aldy Budhi Iskandar, a Data Scientist that just started my data learning journey from January 2022. This dataset is one of the leading banks in Indonesia data with year range from 2007 - 2014.

This dataset consists of **466285 rows** and **75 columns**. In this project, I try to Build most ideal model for predicting Loan Status Customers in dataset by Performing Supervised Classification model Machine Learning.

Contents:

`_rec` :	The total amount committed by investors for that loan at that point in time.

`acc_now_delinq`	: The number of accounts on which the borrower is now delinquent.

`addr_state` :	The state provided by the borrower in the loan application.

`all_util` :	Balance to credit limit on all trades.

`annual_inc`	: The self-reported annual income provided by the borrower during registration.

`annual_inc_joint` :	The combined self-reported annual income provided by the co-borrowers during registration.

`application_type` :	Indicates whether the loan is an individual application or a joint application with two co-borrowers.

`collection_recovery_fee` :	post charge off collection fee.

`collections_12_mths_ex_med` :	Number of collections in 12 months excluding medical collections.

`delinq_2yrs` :	The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
desc	Loan description provided by the borrower.

`dti_joint` :	A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income.

`earliest_cr_line` :	The month the borrower's earliest reported credit line was opened.

`emp_length` :	Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 

`emp_title` :	The job title supplied by the Borrower when applying for the loan.

`Femp` :	A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.

`fico_range_high` :	The upper boundary range the borrower’s FICO at loan origination belongs to.

`fico_range_low` :	The lower boundary range the borrower’s FICO at loan origination belongs to.

`funded_amnt` :	The total amount committed to that loan at that point in time.

`grade` :	LC assigned loan grade.

`home_ownership` :	The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.

`id` :	A unique LC assigned ID for the loan listing.

`il_util` :	Ratio of total current balance to high credit/credit limit on all install acct.

`initial_list_status` :	The initial listing status of the loan. Possible values are – Whole, Fractional.

`inq_fi` :	Number of personal finance inquiries.

`inq_last_12m` :	Number of credit inquiries in past 12 months.

`inq_last_6mths` :	The number of inquiries in past 6 months (excluding auto and mortgage inquiries).

`installment` :	The monthly payment owed by the borrower if the loan originates.

`int_rate` :	Indicates if income was verified by LC, not verified, or if the income source was verified.

`is_inc_v`	: -

`issue_d` :	The month which the loan was funded.

`last_credit_pull_d` :	The most recent month LC pulled credit for this loan.

`last_fico_range_high` :	The upper boundary range the borrower’s last FICO pulled belongs to.

`last_fico_range_low` :	The lower boundary range the borrower’s last FICO pulled belongs to.

`last_pymnt_amnt` :	Last total payment amount received.

`last_pymnt_d` :	Last month payment was received.

`loan_amnt` :	Last month payment was received.

`loan_status` :	Current status of the loan.

`max_bal_bc` :	Maximum current balance owed on all revolving accounts.

`member_id` :	A unique LC assigned Id for the borrower member.

`mths_since_last_delinq` :	The number of months since the borrower's last delinquency.

`mths_since_last_major_derog` :	Months since most recent 90-day or worse rating.

`mths_since_last_record` :	The number of months since the last public record.

`mths_since_rcnt_il`	: Months since most recent installment accounts opened.

`next_pymnt_d` :	Next scheduled payment date.

`open_acc` :	The number of open credit lines in the borrower's credit file.

`open_acc_6m`	: Number of open trades in last 6 months.

`open_il_12m`	: Number of installment accounts opened in past 12 months.

`open_il_24m`	: Number of installment accounts opened in past 24 months.

`open_il_6m` :	Number of installment accounts opened in past 6 months.

`open_rv_12m` :	Number of revolving trades opened in past 12 months.

`open_rv_24m` :	Number of revolving trades opened in past 24 months.

`out_prncp` :	Remaining outstanding principal for total amount funded.

`out_prncp_inv` :	Remaining outstanding principal for portion of total amount funded by investors.

`policy_code` :	"publicly available policy_code=1. new products not publicly available policy_code=2".

`pub_rec` :	Number of derogatory public records.

`purpose` :	A category provided by the borrower for the loan request.

`pymnt_plan` :	Indicates if a payment plan has been put in place for the loan.

`recoveries` :	Indicates if a payment plan has been put in place for the loan.

`revol_bal` :	Total credit revolving balance.

`revol_util` :	Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.     

`sub_grade` :	LC assigned loan subgrade.

`term` :	The number of payments on the loan. Values are in months and can be either 36 or 60.

`title`	: The loan title provided by the borrower.

`tot_coll_amt`	: Total collection amounts ever owed.

`tot_cur_bal`	: Total current balance of all accounts.

`total_acc`	: The total number of credit lines currently in the borrower's credit file.

`total_bal_il` :	Total current balance of all installment accounts.

`total_cu_tl`	: Number of finance trades.

`total_pymnt`	: Payments received to date for total amount funded.

`total_pymnt_inv`	: Payments received to date for portion of total amount funded by investors.

`total_rec_int`	: Interest received to date.

`total_rec_late_fee`	: Late fees received to date.

`total_rec_prncp`	: Principal received to date.

`total_rev_hi_lim` : Total revolving high credit/credit limit.

`url` :	URL for the LC page with listing data.

`verification_status` :	Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified.

`zip_code` :	The first 3 numbers of the zip code provided by the borrower in the loan application.

## **Background & The Steps**
**Background**

* Along with the development of information and technology, it's easier to make loans to banks, without having to meet face-to-face and without having to fill in personal data, we can become prospective loan recipients online.
* However, with this benefit, the bank must also be able to take advantage of these technological advances to facilitate the process of validation and verification of customers who are potentially eligible to receive loans with a good track record.
* Using what methods can the bank do so that it can easily carry out verification in a very easy, fast, and modern way? By creating a machine learning model to predict customer status.
* The company's verification team can apply this model to the verification process of customers who will make loans.

**The steps in this project:**

* Data collection.
* Data understanding by performing the statistic descriptive of data.
* Data preprocessing that includes Handling Missing Value, Data Encoding, and Data Scaling.
* Supervised Classification Modeling.
* Evaluation of the result (best model) base Precision and AUC_ROC to determine the most ideal model by how much it can save banks money from bad track record customers.

## **Conclusion**
From a variety of models, it can be concluded that for this dataset using the Random Forest model is the right choice, because applying the model to this dataset, can save the bank's money from customers who have a bad track record, so the possibility is very small for banks to lose capital due to bad credit customers.

Let's Check out my pdf presentation and python code jupyter notebook!!! Don't hesitate to contact me if you want to do some correction or discussion! aldybudhi003@gmail.com or https://www.linkedin.com/in/aldybudhi/ #DataScience #SupervisedLearning #Classification #Loan #Banking
