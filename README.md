# Counterfeit_Detection


## Project Pitch
Through this project, we have decided to solve two crucial problems considering factors like scalability and depth of solution. 

### Problem 1:
#### Fradulent and Bogus Reviews:
It is estimated that about 11-15% of reviews on ecommerce websites are fraudulent reviews, and in regards to amazon the numbers stands around 40%, they stem from various factors such as bot reviews, review swapping, purchased/incentivized reviews, etc.


To tackle this problem, we have created a **scoring system** which hinges on the following parameters: 
1) Verification Check
2) An **Auto-encoder** which detects anomalies in a review
3) A **Classifier** which is trained on a dataset of labelled reviews (source: https://osf.io/3vds7).
4) **Sentiment Analysis**
5) Miscelleneous parameters like whether a review was tagged as helpful.


Each of these parameters has been assigned a ** Weight ** (which can be changed as per the requirement) and based on this weight, a mathematical formula is calclulated which assigns a score to each review. A higher score signifies a higher tendancy of the review to be fake. 


### Problem 2:
#### Counterfeit/Fake Product Detection:
The proliferation of counterfeit products on Amazon's marketplace poses a significant threat to consumer trust, brand integrity, and fair market competition.

Similar to how we solved Problem 1, we chose specific parameters which define a scoring system. These parameters are:
1) Average review score for a particular product which is taken from the previous problem.
2) Price reletive to other products of the same category
3) Quality of the listing (Readability, richness, detail, etc.
4) Sentiment Analysis

The models have been coded on python and have been integrated with a React.js frontend through Flask. To create the ML and NLP models, sci-kit learn, tensorflow and NLTK has been used.

The **USP** of our product is the extensive analysis giving an accurate score. The project is easy to scale and this can be achieved by using pre-trained models for image analysis, using a better dataset (considering database constraints).
