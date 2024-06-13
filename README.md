# Counterfeit_Detection
Amazon Hackon Project

Scoring parameters:
A) Verification of purchase:
Cross-reference reviews with purchase data to confirm if the reviewer has actually purchased the product.
B) NLP analysis for review similarities and patterns:
Implement advanced NLP techniques to analyze linguistic patterns, sentiment, and uniqueness of the reviews. Detect repetitive or overly similar reviews that may indicate automated or coordinated fake reviews.
C) Classifier for identifying fake reviews:
Develop a machine learning classifier trained on a comprehensive dataset of fake reviews. The classifier will output a probability (x%) of a review being fake. We can constantly update the classifier as we detect fake reviews to improve classifier activity with time.
D) General Account Activity and Purchase History:
Evaluate a user’s account based on different criteria which include frequency and diversity of purchases, review posting behavior (e.g., sudden spikes in activity, patterns of reviewing only certain types of products), length of time for which account has been active, ratio verified-unverified reviews geographic location, return-refund behaviour, etc.
E) Sentiment Analysis
Perform sentiment analysis to detect unusually extreme positive or negative reviews. Use this data to identify potential bias or fraudulent intent.
