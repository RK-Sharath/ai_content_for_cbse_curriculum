## What is Machine Learning

Machine learning is a large field of study that overlaps with and inherits ideas from many related fields such as artificial intelligence. The focus of the field is learning, that is, acquiring skills or knowledge from experience. Most commonly, this means synthesizing useful concepts from historical data. As such, there are many different types of learning that you may encounter as a practitioner in the field of machine learning: from whole fields of study to specific techniques. 

Machine learning (ML) is a subset of Artificial Intelligence (AI) which allows a machine to automatically learn from past data without programming explicitly. We can get Artificial Intelligence system without using machine learning, but this would require building millions of lines of codes with complex rules and decision-trees. All ML is AI but not all AI is ML. Ex :- a chatbot built using rules & complex decision tress is AI but not ML. A chatbot built using ML capabilities would be able to learn new terms and questions and adapt to serve up the most appropriate answers.

#### Types of Machine Learning 

There are three main types of learning in machine learning: supervised, unsupervised, and reinforcement learning.

#### Supervised Learning
Supervised learning describes a class of problem that involves using a model to learn a mapping between input examples and the target variable.

There are two main types of supervised learning problems: they are classification that involves predicting a class label and regression that involves predicting a numerical value.

**`Classification`**: Supervised learning problem that involves predicting a class label. Ex: predict whether an event will happen or not given historical data.

**`Regression`**: Supervised learning problem that involves predicting a numerical label. Ex: predict the temperature for tomorrow given historical data. 

Both classification and regression problems may have one or more input variables and input variables may be any data type, such as numerical or categorical.

#### Unsupervised Learning
Unsupervised learning describes a class of problems that involves using a model to describe or extract relationships in data. Ex : Clustering - group the customer with similar buying patterns.

#### Reinforcement Learning
Reinforcement learning describes a class of problems where an agent operates in an environment and must learn to operate using feedback. Ex: Game theory - player learns from mistakes and corrects them to win the game.


## Machine Learning Model Lifecycle

The lifecycle involves multiple steps per below.

- Understanding the problem : Is it supervised or unsupervised problem?
- Data exploration          : Does the data have any missing values?
- Hypothesis formation.     : Initial assumptions
- Data curation.            : Clean the data by treating missing values, outliers etc
- Experimenting with different techniques : Build & train the model
- Communicating results     : Storytelling to stake holders
- Iterating to achieve objective : Incorporate the feedback from stake holders if any and iterate the previous steps.

## Types of Data

**`Structured Data`** :- Structured data is data that has been organized into a formatted repository, typically a database, so that its elements can be made addressable for more effective processing and analysis. In simple words, the datasets which has rows and column headers qualifies as structured data. Ex :- Names, dates, address, location, product details etc stored in relational databases. 

**`Unstructured Data`** :- Unstructured data is the data which does not conforms to a data model and does not have easily identifiable structure such that it can not be used by a computer program easily. Unstructured data is not organised in a pre-defined manner or does not have a pre-defined data model, thus it is not a good fit for a mainstream relational database. In simple words, the datasets which does not fit into rows and columns qualifies as unstructured data. Ex :- Images, news articles, web pages, reports, surveys. 

**`Semi-structured Data`** :- Semi-structured data is the data which does not conforms to a data model but has some structure. It lacks a fixed or rigid schema. It is the data that does not reside in a rational database but that have some organisational properties that make it easier to analyse. With some process, we can store them in the relational database. In simple words, the datasets which has a mix of structured format and unstructured format qualifies as semi-structured data. Ex :- PDF documents, JSON files and XML documents.

## Natural Language Processing

Natural Language Processing or NLP is a field of Artificial Intelligence that gives the machines the ability to read, understand and derive meaning from human languages. We will understand more about NLP in detail by starting from basics and move ahead to solve an usecase using IBM technology. Ex :- Understand sentiments from text, process and summarize text etc.

[Basics of NLP](https://github.com/RK-Sharath/NLP)

[Click here to know more about NLP](https://github.com/IBM/text-summarization-and-visualization-using-watson-studio/blob/master/CBSE-README.md)

## Data exploration & visualization

Data Exploration is about describing the data by means of statistical and visualization techniques. We explore data in order to bring important aspects of that data into focus like missing data, outliers, correlation & trend for further analysis.This process isn’t meant to reveal every bit of information a dataset holds, but rather to help create a broad picture of important trends and major points to study in greater detail. Data exploration can use a combination of manual methods and automated tools such as data visualizations, and charts.

## Data re-sampling techniques

During data exploration we might realize that the data is highly skewed. Imbalanced data typically refers to a classification problem where the number of observations per class is not equally distributed; often you'll have a large amount of data/observations for one class (referred to as the majority class), and much fewer observations for one or more other classes (referred to as the minority classes). For example, suppose you're building a classifier to classify a credit card transaction a fraudulent or authentic - you'll likely have 10,000 authentic transactions for every 1 fraudulent transaction, that's quite an imbalance! This can be corrected using data resampling techniques such as SMOTE.

## Explore different algorithms and the math behind it

The next step is to explore different algorithms for best fit and the math behind it. This is very important to understand the behaviour of predictive models/algorithms on the corresponding datasets. Its best practice to try atleast three different algorithms on the given dataset to evaluate the performance.

[Click here to explore, visualize, resample data and algorithms to understand the math behind it](https://github.com/IBM/xgboost-smote-detect-fraud/blob/master/CBSE-README.md)


## Build, deploy and retrain the model 

There are three stages in the model building process. They are training, testing & validating the model. First we use the training dataset to enable machine to learn from the data and capture the trends and relationship between variables. After the model is trained, we need to test the model using testing data to evaluate the performance on similar data. Validation data will be used to test the model to evaluate the performance on the generic data to understand how well it can predict on any given datasets.

### Training Dataset: 

The sample of data used to fit the model. The actual dataset that we use to train the model. The model sees and learns from this data.

### Validation Dataset: 

The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.

### Test Dataset: 

The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.

An example of how to create a binary classification model under supervised learning is given below. In this code pattern, we demonstrate how to build a binary classification predictive model using automated and manual methods, train and deploy it to production to identify fraudulent transactions.

[Click here to build, deploy and retrain model](https://github.com/IBM/predict-fraud-using-auto-ai/blob/master/CBSE-README.md)


## Build a Neural Network model from scratch

An example of how to build a neural network model using RNN technique is given below. In this code pattern, we demonstrate different aspects of building a regression based neural network model to forecast cash demand for ATM vending machines. We will also review different terms which are part of the neural networks technique.

[Click here to build a Neural Network Model](https://github.com/IBM/forecast-demand-for-vending-machines/blob/master/CBSE-README.md)


## Hyperparameters Optimization

In machine learning, a hyperparameter of the model is a parameter whose value is used to control the learning process. By contrast, the values of other parameters are derived via training. In simple words, hyperparameters help the model to understand the data better and enhance the accuracy of the predictions by improving the speed and quality of learning process.

An example of how to modify and optimize the hyperparameters is given in the below URL. In this code pattern, we demonstrate how to build a multi class classification neural network model using CNN technique and optimize hyperparameters for enhanced accuracy. 

[Click here to understand Hyperparameter Optimization using CPU and GPU](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/CBSE-README.md)

## Commonly used platforms to build and run models

**`IBM Cloud Services & Watson Studio`** : Simplify and scale data science across any cloud. Prepare data and build models anywhere, using open source codes or visual modeling. Predict and optimize your business outcomes.
[IBM Watson Studio](https://www.ibm.com/cloud/watson-studio)

**`Tensorflow`** : TensorFlow is a library (and also a platform) created by the team behind Google Brain. It’s an implementation of the ML subdomain called Deep Learning Neural Networks; that is to say, TensorFlow is Google’s take on how to achieve machine learning with neural nets using the technique of deep learning.
[Tensorflow](https://www.tensorflow.org/)

**`Amazon Web Services`**: Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high quality models.
[Amazon Sage Maker](https://aws.amazon.com/machine-learning/accelerate-amazon-sagemaker/)

**`Google Cloud Platform`**: Cloud AutoML is a suite of machine learning products that enables developers with limited machine learning expertise to train high-quality models specific to their business needs.
[Google AutoML](https://cloud.google.com/automl/)

## Takeaway for students

The students will benefit to understand different concepts like data exploration, visualization, resampling techniques, math behind different algorithms, machine learning, deep learning, neural networks, building models, hyperparameters optimization and different platforms available for building models. They can play around using the code patterns to get a practical experience of all the learnings.
