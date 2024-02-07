# Term Deposit Marketing

# Background:
Our objective is to create a robust machine learning system for a small startup, focusing mainly on providing machine learning solutions in the European banking market. Leveraging information coming from call center data, we are looking for ways to improve the success rate for calls made to customers for any product that the clients offer. 

Ultimately, we are designing an ever-evolving machine learning product that offers high success outcomes while providing interpretability for our clients to make informed decisions.

# Data Description:
The dataset originates from the direct marketing initiatives of a European banking institution. The marketing campaign entails contacting customers through phone calls, often making multiple attempts to encourage product subscriptions, specifically for term deposits. Term deposits typically represent short-term financial commitments with maturities spanning from one month to a few years.
  
## Attributes:
        	age : age of customer (numeric)
        	job : type of job (categorical) 
        	marital : marital status (categorical)
        	education (categorical)
        	default: has credit in default? (binary)
        	balance: average yearly balance, in euros (numeric) 
        	housing: has a housing loan? (binary)
        	loan: has personal loan? (binary)
        	contact: contact communication type (categorical) 
        	day: last contact day of the month (numeric)
        	month: last contact month of year (categorical)
        	duration: last contact duration, in seconds (numeric)
        	campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
        	Output (Label): y - has the client subscribed to a term deposit? (binary)

# Goal(s):
* Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
* Success Metric(s): Hit 81% or above accuracy by evaluating with 5-fold cross validation, reporting the average performance score.
* Bonus(es): Find the customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize. Find out What makes the customers buy? - Which feature should be the focuse be on.

# Solution:
The dataset has no duplicates or null values. It comprises 40,000 records with a variety of attributes. Notably, the dataset includes both binary and categorical attributes, providing a diverse set of information for analysis.

![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/46b00387-1455-4ce6-8bc9-8e801cff4aa9)

## Following the initial data preprocessing and EDA, the key findings are as follows:

* The data exhibits a noticeable class imbalance, with a prevalent 'No' response, signifying that only a very small percentage of customers belong to the positive class within our dataset.
  ![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/f341cbda-03f3-4891-8d5b-70a318b59b58)
  
* A significant portion of customers were engaged over calls for a duration exceeding 200 minutes. Additionally, the majority of customers were contacted only 1-2 times. Furthermore, a prevailing trend indicates that most customers maintain a relatively low average annual balance.
  ![Untitled1](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/17f33ce7-4565-471b-acbc-f62cf7d77b4e)

* The dataset contains outliers, particularly in the 'duration' and 'balance' attributes. We will retain outliers, given the imbalanced class nature. Additionally, normalization is skipped as it resulted in reduced model accuracy.

![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/fab921d0-6fdd-4b47-846c-038480afd7ff)


We performed one-hot encoding on all the categorical variables, converting them into a format suitable for machine learning algorithms.

![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/6c606723-ac1e-4854-8b16-7eba84c9df03)

* From the correlation analysis, it is observed that the variable "duration" exhibits the highest positive correlation, with a coefficient of 0.46.
  ![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/18b674d8-5634-4b18-a917-5dfe86e95fc5)

# Building and training Models
## Oversampling:
We conducted oversampling on the training split to address the imbalanced classes. Prior to oversampling, the training set had 1741 instances in the positive class and 22,259 instances in the negative class. After oversampling, the training set counts were balanced, with both the positive and negative classes having 22,259 instances each.

## Model Crossvalidation (SKF with 5 Fold):
We employed a 5-fold cross-validation using Stratified K-Fold (SKF) for model evaluation using training split. This technique divides the dataset into five folds, ensuring that each fold maintains the same distribution of classes as the original dataset, providing a robust evaluation of the model's performance.

![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/a3dcbebd-fcd0-4703-baed-5de42361eac3)

## Model Evaluation using Validation Split:
![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/b69fabe7-8c34-4c12-98ba-af93bfe5dddb)

## Final Model Evaluation on test dataset:
After evaluating the performance on the validation split, we have opted for the **RandomForestClassifier** as our final model, given its notable F1 score of 0.94.

### Classification report for the RandomForestClassifier on the test split:
![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/11b1b463-7132-447d-812c-2c0c2d0cb8e8)
## Observations:
* Across all our models, we have achieved a success metric of over 81% accuracy. The RandomForestClassifier stands out with an impressive accuracy of 93% on the test dataset and 93.6% on the validation dataset.
* Employing a 5-fold Stratified K-Fold (SKF) cross-validation, we attained an average accuracy score of 98.3 on our training dataset.
* While accuracy serves as a valuable measure of model performance, we acknowledge its limitations as the sole metric. However, for our specific objectives, accuracy has been chosen as the primary success indicator.
* Notably, our classes exhibited a significant imbalance, leading us to implement a random oversampler on the training dataset. This strategy aimed to balance the class distribution during the training of our models.

# Feature Importance:
SHAP values offer more detailed insights into the factors influencing a customer's decision to subscribe to a term deposit.
* **Positive Contributing Features**: The SHAP plot presented here specifically highlights features that positively impact predicting the customer's outcome. These factors contribute to higher chances of a customer subscribing to a term deposit.
* **Duration of Calls**: Consistent with earlier analyses, this SHAP value plot reinforces the critical role of call duration in achieving positive customer outcomes. Notably, longer call durations have an even more significant impact on the likelihood of a positive customer outcome, as indicated by prominent red highlights in the plot.
* **Contact with Unknown Parties**: Another noteworthy feature insight is that the "contact_unknown" feature ranks as the second most important factor influencing a positive model output. This implies that engaging with unknown contacts is correlated with an increased likelihood of customer subscriptions. In essence, reaching out to new or previously uncontacted customers appears to be beneficial for achieving positive outcomes.

![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/24687ff5-a442-4541-bb21-f640f400d12b)
 *(Note: It's crucial to recognize that the observations from the SHAP values are limited by the data used for their generation. These values were computed based on the test data split, and the analysis focused exclusively on positive outcomes. Therefore, while these insights offer valuable information, their interpretation should consider the context of the specific dataset and analysis conducted.)*

### What makes the customers buy? Which feature we should be focusing more on?
* Based on the insights provided by the plot and table above, it is evident that **"Duration"** stands out as a crucial feature for predicting customer outcomes.
* When prioritizing the focus on these features, it is advisable to start with "Duration" - the duration of calls made to the customer. Subsequently, "Balance," representing the average yearly balance, should receive attention. Additionally, the day of the month on which the customer is contacted and the customer's age are also significant and warrant focus in that order.

# Customer Segmentation:
##Finding customers who are more likely to buy the investment product. Determine the segment(s) of customers the client should prioritize.

![image](https://github.com/kuzhuppillil/d8Ps30XsTi68sD6W/assets/25860818/559942bc-db6b-4c8c-923e-de53eaafc4d2)

### Overall Segment Properties
* Customers are grouped into three segments based on predicted probability scores: low, medium, and high.
* The low segment comprises approximately 70% of the customers, the medium segment includes around 28%, and the high segment contains roughly 10%.
* The high and medium segments stand out as the most valuable customer groups with a higher likelihood of subscribing to term deposits.

### Low Segment
* The low segment exhibits the lowest predicted subscription likelihood.
* Customers in this segment tend to have shorter last contact durations, averaging around 574 seconds.
* They maintain lower average yearly account balances, approximately 518 euros.
* Typically, customers in this segment are middle-aged, with an average age of 39 years.
* A significant portion of customers in this segment has existing housing loans.

### Medium Segment
* The medium segment represents a moderate predicted subscription likelihood.
* Customers in this group engage in medium-length last contact durations, averaging around 809 seconds
* They maintain moderate yearly account balances, averaging about 754 euros.
* The average age of customers in this segment is around 40 years.
* Most customers in this segment do not have housing loans.

### High Segment
* The high segment stands out with the highest predicted subscription likelihood.
* Customers in this group have longer last contact durations, averaging approximately 1,205 seconds.
* They maintain higher yearly account balances, with an average of 990 euros.
* The average age of customers in this segment is around 41 years.
* Unlike the low segment, most customers in this segment do not have housing loans.

### Prioritizing Customers
The high and medium segments, comprising middle-aged, high-net-worth customers without housing loans, should be the primary target for marketing term deposits. These customer segments are more likely to subscribe and are willing to spend a considerable amount of time during contact, which is a key indicator of their likelihood to subscribe. Thus, prioritizing marketing efforts towards these segments, limiting campaigns to no more than two, and scheduling campaigns around the middle of the month during the first quarter of the year can be an effective strategy. This targeted approach aims to maximize the likelihood of customer subscriptions to term deposits.

  

