![alt text](https://github.com/GVRQ/TS3-DS/blob/main/images/nuwe.jpg?raw=true)
# Influencias en el rendimiento académico
* eng: **Influences on academic performance**

## Background
A study has been carried out to see if the academic performance of children is influenced by the academic level of their parents. Therefore, the academic results of the students will be evaluated based on several variables.


## *Problem*
To find a solution to a multi-class classification problem we have two datasets. One of them is Train, that contains student's data and data on students' parental educational level. The second dataset is Test, that contains no data on parental educational level.

We will perform EDA and create a ML model for prediction of parental educational level on Test-dataset.
-  Predictive model using ***GaussianNB***. 
- We have evaluated over 20 basic models and come to conclusion that Gaussian Naive Bayes has the best results.

A dataset of 800 rows for the training [(train)](https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Talent+Squad+League/3rd_batch/data/train.csv) of the prediction algorithm and 200 for the testing [(test)](https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Talent+Squad+League/3rd_batch/data/test.csv).

## *Variables:*
- `gender` : student`s gender

- `parental level of education`: educational level of the parents

- `lunch`: school lunch

- `test preparation course`: attend the prep course

- `math score`: Math score

- `reading score`: Reading score

- `writing score`: Writing score

Numbers represent following parental educational level:
- high school: 0,

- some high school: 1,

- some college: 2,

- associate's degree: 3,

- bachelor's degree: 4,

- master's degree: 5

## *Goal*
1. The goal of the challenge is to provide an answer to whether the student's academic results are influenced by the educational level of the parents. 
2. Create a predictive model for prediction of test-dataset.

## Results

1. The results of the first goal are provided in the Final Conclusion of [(TS3-DS.ipynb-file)](https://github.com/GVRQ/TS3-DS/blob/main/TS_DS3.ipynb). In short: The student's academic results are **influenced by the educational level of the parents**.
2. The results of the parental educational level are in the [(predictions.csv-file)](https://github.com/GVRQ/TS3-DS/blob/main/predictions.json).

## Analysis
We've analyzed the data and come to conclusion that children from families with higher educational levels tend to score better in all areas. 
![alt text](https://github.com/GVRQ/TS3-DS/blob/main/images/Parental_ed_values.png?raw=true)

However **the parental educational level is not the key factor for the students performance**. Students that completed **Test preparation course** achieved higher results than students that haven't completed the prep course.
![alt text](https://github.com/GVRQ/TS3-DS/blob/main/images/prep_course_vs_scores.png?raw=true)

# Solution
After analyzing Correlations between features we've dropped `high_income` feature from datasets.

**Model: GaussianNB without optimizations.**
The best results obtained with the selected model.

- **Accuracy: 0.2917**
- **F1-Score macro: 0.2608**
- **F1-Score micro: 0.2917**
- **F1-Score weighted: 0.2765**
![alt text](https://github.com/GVRQ/TS3-DS/blob/main/images/ML_results.png?raw=true)

## License
The open source license. https://opensource.org/licenses/MIT
MIT License

Copyright (c) 2022 Alexander Gavrilov

<h3 style="text-align: left;" align="left">Connect with me:</h3>
<p style="text-align: left;" align="left"><a href="https://t.me/gavrilov_se" target="blank"><img style="float: left;" src="https://www.svgrepo.com/show/349527/telegram.svg" alt="Telegram_Alexander_Gavrilov_Data_Scientist" width="40" height="30" align="center" /></a>&nbsp;<a href="mailto:alexander@gavrilov.se" target="blank"><img src="https://www.clipartmax.com/png/full/91-913506_computer-icons-email-address-clip-art-icon-email-vector-png.png" alt="Email_Alexander_Gavrilov_Data_Scientist" width="30" height="30" align="center" /></a>&nbsp; <a href="https://www.linkedin.com/in/GVRQ/" target="blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/72px-LinkedIn_icon.svg.png" alt="Linkedin_Alexander_Gavrilov_Data_Scientist" width="30" height="30" align="center" /></a></p>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

### Improvements 💡
There are several ways to improve this model, including:
1.    Using more and better features: The current model only uses a few features (e.g. parental education, test preparation, lunch type) to predict the overall performance of a student. Adding more relevant features, such as the student's gender, age, and socioeconomic status, could potentially improve the model's performance.

2.    Using more advanced machine learning algorithms: The current model uses a simple Gaussian naive Bayes classifier, which may not be the most appropriate algorithm for this problem. Using more advanced algorithms, such as decision trees, random forests, or support vector machines, could potentially improve the model's performance.

3.    Using hyperparameter tuning: The current model does not use any hyperparameter tuning, which means that the model's performance may not be optimized. Using techniques like grid search or random search to find the best hyperparameters for the model could potentially improve its performance.

4.    Using more data: The current model uses a relatively small amount of data, which may not be enough to train a high-performance model. Using more data, either by collecting more data or using techniques like data augmentation, could potentially improve the model's performance.

5.    Evaluating the model's performance more thoroughly: The current model only uses a few metrics (e.g. F1, accuracy, precision, recall) to evaluate its performance. Using more comprehensive evaluation metrics, such as receiver operating characteristic (ROC) curves, could provide a more thorough understanding of the model's performance.
