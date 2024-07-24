
### Note:
1) The option to leave a custom review on the product listing and the website hosting will be incorporated by the next round.
2) On our homepage, a red border around the product indicates that the product has been flagged as bogus, according to our model.
3) Our algorithm (including all the ML models) has been trained on real-world datasets and is considerably accurate. (Outputs low review score for authentic amazon products)
4) The website is optimised to run on devices of all aspect ratios.

## Project Pitch
Through this project, we have decided to solve two crucial problems considering factors like scalability and depth of solution. 

### Problem 1:
#### Fraudulent and Bogus Reviews:
It is estimated that about 11-15% of reviews on e-commerce websites are fraudulent reviews, and in regards to Amazon, the numbers stand around 40%; they stem from various factors such as bot reviews, review swapping, purchased/incentivized reviews, etc.


To tackle this problem, we have created a **scoring system** which hinges on the following parameters: 
1) Verification Check
2) An **Auto-encoder** which detects anomalies in a review
3) A **Classifier**, which is trained on a dataset of labelled reviews.
4) **Sentiment Analysis**
5) Miscellaneous parameters like whether a review was tagged as helpful.


Each of these parameters has been assigned a ** Weight ** (which can be changed as per the requirement), and based on this weight, a mathematical formula is calculated, which assigns a score to each review. A higher score signifies a higher tendency of the review to be fake. 


### Problem 2:
#### Counterfeit/Fake Product Detection:
The proliferation of counterfeit products on Amazon's marketplace poses a significant threat to consumer trust, brand integrity, and fair market competition.

Similar to how we solved Problem 1, we chose specific parameters which define a scoring system. These parameters are:
1) Average review score for a particular product which is taken from the previous problem.
2) Price reletive to other products of the same category
3) Quality of the listing (Readability, richness, detail, etc.)
4) Sentiment Analysis

The models have been coded on python and have been integrated with a React.js frontend through Flask. To create the ML and NLP models, sci-kit learn, tensorflow and NLTK has been used.

The **USP** of our product is the extensive analysis giving an accurate score. The project is easy to scale and this can be achieved by using pre-trained models for image analysis, using a better dataset (considering database constraints).

The UI of our frontend depicts the ease of integration into pre-exisitng amazon services.

## File Types:
1) Main.py: Contains the actual machine learning models and all the data frame used in the models.
2) transfromer_pipeline.py: This contains a prebuilt transformer-pipeline not currently used due to token constraints. Added to demonstrate **scalability** of project.
3) backend: Contains our backend server.
4) website/amazon: Contains our frontend files.
5) Csv files:
    - a) Classifer_dataset.csv: CSV file used to train the classifier.(See problem 1, paramter 3 in main.py)
              (source: https://osf.io/3vds7)
    - b) final_dataset.csv: Main Dataset used for obtaining the reviews.
    - c) product.csv: Main dataset used for obtaining products. (derived from final_dataset.csv)
    - d) Finalfrontenddb.csv: Dataset used for making the backend node.js server

## How to use:
1) Ensure that node.js and npm are installed and can be used from the cmd/terminal.
2) Clone the repository and navigate to the project location in the cmd/terminal.
3) To run the Python script (assuming you have Python installed), in cmd/terminal run ```pip install -r requirements.txt```
4) In a new terminal, navigate to the backend folder and run ```node server.js```
5) From the counterfeit_Detection folder, navigate to the amazon folder (```cd ./website/amazon```) and in terminal run ```npm start```


## Images

<img width="1800" alt="Screenshot_2024-07-01_at_10 43 24_PM" src="https://github.com/Snimso/Counterfeit_Detection/assets/115739027/cfab7019-a637-46e1-a542-c3b683128ca3">

<img width="1800" alt="Screenshot_2024-07-01_at_10 43 26_PM" src="https://github.com/Snimso/Counterfeit_Detection/assets/115739027/2be7cba1-0cda-431e-86e5-a06a35763561">

<img width="1800" alt="Screenshot_2024-07-01_at_10 43 29_PM" src="https://github.com/Snimso/Counterfeit_Detection/assets/115739027/a03ba29c-55ea-4eaa-ac33-303dc8bebbc9">

<img width="1800" alt="Screenshot_2024-07-01_at_10 43 32_PM" src="https://github.com/Snimso/Counterfeit_Detection/assets/115739027/4618538b-bf87-4186-b424-1564066e3fe9">

<img width="1800" alt="Screenshot_2024-07-01_at_10 43 41_PM" src="https://github.com/Snimso/Counterfeit_Detection/assets/115739027/15e1c932-2322-4c9d-9413-9e2cdf6bdf65">
