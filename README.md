# <div align="center">Analysis of Sleep Duration and its Influencing Factors Using MLR</div>
## <div align="center">![Intro](images/sleep.jpg)

## Problem Statement
The purpose of this analysis is to understand the relationships between various health and lifestyle factors and sleep duration. Specifically, we seek to identify which variables (such as age, heart rate, physical activity level, BMI category, and occupation) are most predictive of sleep duration in a sample population. Using multiple linear regression (MLR), we aim to build an optimal model to predict sleep duration based on these variables and evaluate the model's accuracy and assumptions.

## Data Description
The source of the data is from [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

### **Columns Explanation:**
- Person ID: *An identifier for each individual.*
- Gender: *The gender of the person (Male/Female).*
- Age: *The age of the person in years.*
- Occupation: *The occupation or profession of the person.*
- Sleep Duration (hours): *The number of hours the person sleeps per day.*
- Quality of Sleep (scale: 1-10): *A subjective rating of the quality of sleep, ranging from 1 to 10.*
- Physical Activity Level (minutes/day): *The number of minutes the person engages in physical activity daily.*
- Stress Level (scale: 1-10): *A subjective rating of the stress level experienced by the person, ranging from 1 to 10.*
- BMI Category: *The BMI category of the person (e.g., Underweight, Normal, Overweight).*
- Blood Pressure (systolic/diastolic): *The blood pressure measurement of the person, indicated as systolic pressure over diastolic pressure.*
- Heart Rate (bpm): *The resting heart rate of the person in beats per minute.*
- Daily Steps: *The number of steps the person takes per day.*
- Sleep Disorder: *The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea).*
### **Details about Sleep Disorder Column:**
- None: *The individual does not exhibit any specific sleep disorder.*
- Insomnia: *The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.*
- Sleep Apnea: *The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.*

The target variable (dependent variable) should be continuous, so I chose **Sleep Duration** over Quality of Sleep as the target. This is because Quality of Sleep is rated on a 1–10 discrete ordinal scale, which may be better suited for ordinal regression.

## **The Models:**
### 1. Multiple Linear Regression (MLR) 📊 
MLR is included all potential predictors, yielding a summary of each predictor’s statistical significance (p-values) and multicollinearity through Variance Inflation Factor (VIF) values. 

![mlr](images/MLR.png)

**Insights:**

This model explains 88.5% of the variation in sleep duration, with most factors significantly influencing it, while a few, like Heart Rate and certain BMI categories, show little to no impact.

### 2. Best MLR Model 🏆 📊 ➕ 📈 ➔ 🎯
Through variable selection techniques (forward selection, backward elimination, and stepwise selection), we identified an optimal subset of predictors. Stepwise selection was chosen as the preferred method, balancing model simplicity with predictive accuracy. The final selected predictors were:
- Heart Rate
- Physical Activity Level
- Occupation_Engineering/IT
- Daily Steps
- Age
- MAP (Mean Arterial Pressure)
- Occupation_Healthcare
- Occupation_Legal
- BMI Category_Obese
- Sleep Disorder_Sleep Apnea
- Occupation_Sales
  
**Model Performance:**

- R-Squared: 0.8086
- Mean Squared Error (MSE): 0.1209

