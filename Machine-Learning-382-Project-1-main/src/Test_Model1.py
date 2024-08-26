from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from prepare_data import prepare_data
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier


df = prepare_data('../MLG382 Projects/Machine-Learning-382-Project-1/data/raw_data.csv')
X = df.drop(['Loan_Status'], axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_model():
    model = Sequential([
        Dense(21, activation='relu', input_shape=(21,)),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=25, batch_size=32, verbose=0)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_val_score(model, X_train_scaled, y_train, cv=kfold)

print("Cross-validation Accuracy: %.2f%% (+/- %.2f%%)" % (results.mean() * 100, results.std() * 100))