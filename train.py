import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocess import Preprocessor, load_csv

DATA_PATH = 'sample_data.csv'

def prepare_labels(df):
    labels = df['career_label'].astype('category')
    y = labels.cat.codes.values
    label_map = dict(enumerate(labels.cat.categories))
    return y, label_map

def build_nn(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    df = load_csv(DATA_PATH)
    pre = Preprocessor()
    pre.fit(df)
    X = pre.transform(df)
    y, label_map = prepare_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('RF Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(rf, 'rf_model.joblib')

    n_classes = len(label_map)
    nn = build_nn(X_train.shape[1], n_classes)
    nn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    nn.save('nn_model')

    rf_proba = rf.predict_proba(X_test)
    nn_proba = nn.predict(X_test)
    ensemble_proba = (rf_proba + nn_proba) / 2.0
    ensemble_pred = ensemble_proba.argmax(axis=1)

    print('Ensemble Accuracy:', accuracy_score(y_test, ensemble_pred))
    print('Top-3 Accuracy:', top_k_accuracy_score(y_test, ensemble_proba, k=3))
    print(classification_report(y_test, ensemble_pred))

    joblib.dump(pre, 'preprocessor.joblib')
    joblib.dump(label_map, 'label_map.joblib')

if __name__ == '__main__':
    main()
