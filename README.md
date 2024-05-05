# AI Project Report.

# Introduction

The project JDMRevs is dedicated to enhancing personalized automotive experiences within the JDM car enthusiast community. It endeavors to provide a platform for users to select and optimize them according to individual preferences. Through a blend of advanced technological features, such as AI-driven recommendations and real-time visualizations, JDMRevs aims to revolutionize the car modification landscape.
Technologies and ML, DL, NLP Models are used in our project like Linear Regression, Decsion Trees, Random Forest, Cosine Similarity etc.

# Dataset

The dataset used for this project consists of extremely important cars information obtained from kaggle as well as scraping (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). It includes features such as car_name, Brand, selling_price, Engine, Torque, mileage, fuel, km_driven,seats and owner etc. This dataset looks like this:

This is pairplot graph of all attributes.

![1](https://github.com/TahaSaqib1/AI/assets/113784961/a3548205-a9e9-40d2-911c-2684f1909081)

This is HeatMap corr, which will tell us about the correlation of different attributes

![2](https://github.com/TahaSaqib1/AI/assets/113784961/8ca2fc0b-eedd-421d-9ada-67549f341755)

This graph tells about the Brands, their descriptions and occurences

![3](https://github.com/TahaSaqib1/AI/assets/113784961/7dad8272-7739-4446-8b50-70e395049e4d)

This graph tells about the car_names, their descriptions and occurences

![4](https://github.com/TahaSaqib1/AI/assets/113784961/d8e8c2e6-37f3-46e9-9b8a-3eb0e2de0e08)

# Preprocessing

Before applying algorithms and proceedings, we have done with preprocessings:
- Data Loading and Inspection:
  - Loaded the dataset using pandas' read_csv() function.
  - Inspected the first few rows of the dataset using head() to understand its structure and contents.
  - Checked the data types and non-null values in each column using info() to assess data quality.
  - Examined the shape of the dataset using shape to determine the number of rows and columns.
    
- Data Cleaning:
  - Removed unnecessary columns such as 'seller_type' using drop() to streamline the dataset.
  - Handled missing values by either dropping them using dropna() or filling them with column means to ensure data completeness.
  - Eliminated duplicate rows with drop_duplicates() to avoid redundancy in the dataset.
    
- Data Encoding:
  - Identified categorical columns using list comprehensions based on data types.
  - Utilized LabelEncoder from sklearn.preprocessing to transform categorical features into numerical values.
    
- Model Training:
  - Split the dataset into training and testing sets using train_test_split() to evaluate model performance.
  - Trained regression models such as Linear Regression, Decision Tree, Random Forest, and Gradient Boosting using the training data.
    
- Model Evaluation:
  - Evaluated model performance using metrics such as Explained Variance Score and R-Square Score to gauge accuracy and variance explanation capabilities.
    
These preprocessing steps collectively ensure that the dataset is clean, properly formatted, and suitable for training predictive models to predict car_names accurately.

After that, Models are performed, evaluated and then further workings performed.

![5](https://github.com/TahaSaqib1/AI/assets/113784961/4421c3a0-1756-4b65-94ad-8b09c198da16)

# ML Models Evaluation

Models evaluation has been done through Accuracy, Variance, Mean Squared Error and Root Mean Square Error. The models included are Linear Regression, Decision Trees, Gradient Boosting and Random Forest etc.
Results Provided are: 

![ME](https://github.com/TahaSaqib1/AI/assets/113784961/73da7fe9-0e0f-4ab7-b352-f736a3da8c25)

![mse](https://github.com/TahaSaqib1/AI/assets/113784961/c4df2035-199b-4b56-9d1b-a15e872d1d7d)

# Deep Learning Models
The following deep learning models are implemented:
1) ANN:
   
   ![ann train](https://github.com/TahaSaqib1/AI/assets/113784961/32fa1d8e-1df4-44c7-a48d-4adf15d27fea)
   

2) RNN:
   
   ![rnn train](https://github.com/TahaSaqib1/AI/assets/113784961/05883c02-e8d7-487d-8866-3f77a8272c8e)
   

# DL Models Evaluation
1) ANN:
   
   ![1111](https://github.com/TahaSaqib1/AI/assets/113784961/54f53af5-729f-48e7-8e61-a9b53f50f0ad)

   ![ann](https://github.com/TahaSaqib1/AI/assets/113784961/d78f513d-3c8e-445b-b6e7-ff1832c7a1aa)
   
2) RNN:
   
   ![rnn](https://github.com/TahaSaqib1/AI/assets/113784961/592566b8-5949-4dc7-a81a-4f8a4293aa53)

   ![rnn](https://github.com/TahaSaqib1/AI/assets/113784961/928f5420-a699-47cf-99b9-bf2fe71f2424)


# User Interface For ANN, RNN, Cosine Similarity

As a portal for recommendation and optimization through user interface. Here is the provided one:

The user inputs values to specify car preferences: they select "Ford" as the brand, diesel as fuel type, manual transmission, a vehicle age requirement of 7 years, and a maximum price range of 100,000. The program then predicts three cars using different algorithms: ANN suggests "Mahindra Marazzo," RNN suggests "Mahindra XUV300," and Cosine Similarity suggests "Hyundai Venue" as suitable options.


![final](https://github.com/TahaSaqib1/AI/assets/113784961/4c0fed16-866e-46d5-9dcd-839575a9a3e9)


# Conclusion

The project employed ANN, RNN, and cosine similarity models to predict car names based on user preferences like brand, fuel type, and transmission. ANN predicted 'Mahindra Marazzo', while RNN suggested 'Mahindra XUV300'. Cosine similarity recommended 'Hyundai Venue', showcasing diverse approaches to car prediction. These models offer valuable insights for efficient car selection.
The project aimed to predict car names using ANN, RNN, and cosine similarity models. ANN achieved the highest accuracy, followed by RNN and cosine similarity (Early-Stopping applied).

# More Info

For further details and implementation, please refer to the code repository on GitHub.

Github Repository Link: https://github.com/TahaSaqib1/AI

Author: Taha and Tayyab
Date: 4th May, 2024
