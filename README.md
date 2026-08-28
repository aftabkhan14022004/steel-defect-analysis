STEEL SURFACE DEFECT ANALYSIS 🔩
A Two-Phase Industrial Quality Analytics & Deep Learning Project
Combining real-world manufacturing data with SQL, statistics, and computer vision to answer one question: What drives steel surface defects?

📊 PROJECT OVERVIEW
This project investigates quality drivers in a hot-strip steel mill through two complementary approaches:

Phase 1: Data Analytics & SQL
Analyzed 1,800 real steel defect images linked with 5,000 synthetic production batches to test whether process parameters (shift, temperature, speed) affect defect occurrence.

Phase 2: Deep Learning & Computer Vision
Built a CNN classifier using transfer learning to identify steel surface defects directly from images — achieving 99.17% test accuracy.

🔍 PHASE 1: SQL ANALYSIS & STATISTICAL TESTING
The Dataset
Source	Description
NEU Steel Surface Defect Database	1,800 real grayscale images, 6 defect classes
Synthetic Production Data	5,000 batches with furnace temp, rolling speed, shift
MySQL Database	3 normalized tables with foreign keys
Defect Types Analyzed
Crazing | Inclusion | Patches | Pitted Surface | Rolled-in Scale | Scratches

Realistic Data Challenges Introduced
2% missing furnace temperatures (sensor failures)

5 duplicate batch IDs (double-entry errors)

Mixed timestamp formats (system inconsistencies)

What I Did
Designed 3-table relational database with referential integrity

Wrote advanced SQL queries: CTEs, window functions, subqueries

Cleaned messy data: handled nulls, duplicates, inconsistent dates

Created 7 EDA visualizations including probability heatmaps

Ran formal statistical tests with honest interpretation

Results
Hypothesis	Test	p-value	Verdict
Shift affects defect type	Chi-square	0.126	Not significant
Temperature affects defect rate	Welch's t-test	0.276	Not significant
Speed correlates with defects	Pearson r	0.058	Not significant
Key Insight: No single process parameter showed statistical significance — proving the importance of rigorous testing before assuming causal relationships.

🤖 PHASE 2: CNN IMAGE CLASSIFIER
Why Deep Learning?
When tabular data showed no clear signal, the images themselves held the answer.

Model Architecture
Input (224×224×3) → MobileNetV2 (Pre-trained, Frozen) → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(6, Softmax)

Training Results
Metric	Score
Training Accuracy	99.93%
Validation Accuracy	98.89%
Test Accuracy	99.17%
F1-Score (Macro)	0.99
Misclassifications	3 out of 360 images
Note: The NEU dataset has visually distinct defect classes and is well-suited for transfer learning. Published research on this dataset reports accuracies of 95–100%. This result is consistent with existing literature.

Key Files
File	Purpose
data_prep.py	Image loading and preprocessing
train_model.py	Transfer learning model training
evaluate.py	Metrics and confusion matrix
predict.py	Single image prediction
best_model.h5	Trained model
📁 PROJECT STRUCTURE
text
steel-defect-analysis/
├── Phase1_SQL_Analysis/
│   ├── scripts/
│   │   ├── generate_data.py
│   │   ├── load_neu_data.py
│   │   ├── eda.py
│   │   └── statistical_tests.py
│   ├── sql/
│   │   ├── schema.sql
│   │   └── analysis_queries.sql
│   └── fig/
├── Phase2_CNN_Classifier/
│   ├── data_prep.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── best_model.h5
│   └── confusion_matrix.png
├── README.md
├── requirements.txt
└── .gitignore
🚀 SETUP INSTRUCTIONS
Prerequisites
Python 3.11

MySQL (XAMPP)

Git

Quick Start
git clone https://github.com/aftabkhan14022004/steel-defect-analysis.git
cd steel-defect-analysis
pip install -r requirements.txt

Phase 1 Setup
Create MySQL database steel_defects

Import schema: Phase1_SQL_Analysis/sql/schema.sql

Download NEU dataset from Kaggle

Place NEU-DET folder inside data/raw/

Generate and load data:
python Phase1_SQL_Analysis/scripts/generate_data.py
python Phase1_SQL_Analysis/scripts/load_neu_data.py

Run analysis:
python Phase1_SQL_Analysis/scripts/eda.py
python Phase1_SQL_Analysis/scripts/statistical_tests.py

Phase 2 Setup
Train the model:
python Phase2_CNN_Classifier/train_model.py

Evaluate:
python Phase2_CNN_Classifier/evaluate.py

Predict single image:
python Phase2_CNN_Classifier/predict.py

🛠️ TECH STACK
Phase 1
Python, pandas, NumPy, MySQL, SQL, Matplotlib, Seaborn, SciPy

Phase 2
TensorFlow, Keras, MobileNetV2, scikit-learn, Matplotlib, Seaborn

💡 KEY LEARNINGS
Designed normalized relational database with foreign keys

Wrote production-level SQL (CTEs, window functions, subqueries)

Handled real-world messy data and documented every decision

Ran formal hypothesis tests and interpreted p-values honestly

Built transfer learning CNN with 99.17% test accuracy

Learned when tabular analysis fails and image-based approaches succeed

👤 AUTHOR
Aftab Khan
Email: aftabkhan14022004@gmail.com
LinkedIn: https://www.linkedin.com/in/aftab1402/
GitHub: https://github.com/aftabkhan14022004

If this project helped you, consider giving it a star on GitHub!

Copy everything above. Paste into README.md. Save. Then push:

git add README.md
git commit -m "Final README with accuracy context"
git push origin master

Your project is complete. 🎉

