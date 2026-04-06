import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset
data = pd.read_csv("student_data.csv")

# Step 2: View data (for checking)
print("Dataset Preview:")
print(data.head())

# Step 3: Define input (X) and output (y)
X = data[['study_hours', 'attendance', 'previous_marks', 'sleep_hours']]
y = data['final_marks']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Create model
model = LinearRegression()

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predict on test data
predictions = model.predict(X_test)

# Step 8: Evaluate model
error = mean_squared_error(y_test, predictions)
print("\nModel Error:", error)

# Step 9: Take user input
print("\n--- Enter Student Details ---")
study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance (%): "))
previous_marks = float(input("Previous Marks: "))
sleep_hours = float(input("Sleep Hours: "))

# Step 10: Predict result
result = model.predict([[study_hours, attendance, previous_marks, sleep_hours]])

print("\n Predicted Final Marks:", round(result[0], 2))