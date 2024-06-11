#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ClassificationPerformance:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.conf_matrix = confusion_matrix(self.true_labels, self.predicted_labels)
        self.class_report = classification_report(self.true_labels, self.predicted_labels, output_dict=True)
        
    def plot_confusion_matrix(self):
        plt.figure(figsize=(10, 7))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
    def classification_report_df(self):
        report_df = pd.DataFrame(self.class_report).transpose()
        return report_df

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_labels)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def chi_square_test(self):
        chi2, p, dof, ex = stats.chi2_contingency(self.conf_matrix)
        return {'chi2': chi2, 'p_value': p, 'dof': dof, 'expected': ex}

    def plot_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            plt.figure(figsize=(10, 7))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance')
            plt.show()
        else:
            print("Model does not have feature_importances_ attribute.")

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict the labels
y_pred = model.predict(X_test)

# Evaluate the performance
perf = ClassificationPerformance(y_test, y_pred)
perf.plot_confusion_matrix()
print(perf.classification_report_df())
perf.plot_roc_curve()
print(perf.chi_square_test())
perf.plot_feature_importance(model, [f'feature_{i}' for i in range(X.shape[1])])

#%%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 8000 patients
n_patients = 8000

# Profile variables
gender = np.random.choice(['Male', 'Female'], size=n_patients)
age = np.random.randint(18, 90, size=n_patients)
diagnosis = np.random.choice(['Diagnosis A', 'Diagnosis B', 'Diagnosis C'], size=n_patients)
alcohol = np.random.randint(0, 20, size=n_patients)  # drinks per week
smoking = np.random.choice(['Non-smoker', 'Former smoker', 'Current smoker'], size=n_patients)

# Measurements variables
blood_pressure = np.random.randint(90, 180, size=n_patients)
mean_arterial_pressure = np.random.randint(70, 110, size=n_patients)
pulse = np.random.randint(50, 120, size=n_patients)
oxygen_saturation = np.random.randint(85, 100, size=n_patients)
weight = np.random.randint(50, 150, size=n_patients)  # in kilograms
ventilator_duration = np.random.randint(0, 48, size=n_patients)  # hours on ventilator

# Admission and follow-up variables
icu_duration = np.random.randint(1, 20, size=n_patients)  # days in ICU
death_within_year = np.random.choice([0, 1], size=n_patients, p=[0.9, 0.1])  # 10% mortality rate

# Create a DataFrame
data = pd.DataFrame({
    'Gender': gender,
    'Age': age,
    'Diagnosis': diagnosis,
    'Alcohol': alcohol,
    'Smoking': smoking,
    'Blood Pressure': blood_pressure,
    'Mean Arterial Pressure': mean_arterial_pressure,
    'Pulse': pulse,
    'Oxygen Saturation': oxygen_saturation,
    'Weight': weight,
    'Ventilator Duration': ventilator_duration,
    'ICU Duration': icu_duration,
    'Death Within Year': death_within_year
})

# Create a contingency table
contingency_table = pd.crosstab(data['Gender'], data['Death Within Year'])

# Perform the chi-square test
chi2, p, dof, ex = stats.chi2_contingency(contingency_table)

# Print the results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(ex)

# Visualize the contingency table
contingency_table.plot(kind='bar', stacked=True)
plt.title('Death Within Year by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# %%
