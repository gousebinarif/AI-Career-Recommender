import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
import joblib

def load_csv(path):
    return pd.read_csv(path)

def split_skills_interests(df, skills_col='skills', interests_col='interests'):
    df = df.copy()
    df['skills_list'] = df[skills_col].fillna('').apply(lambda s: [x.strip() for x in s.split(';') if x.strip()])
    df['interests_list'] = df[interests_col].fillna('').apply(lambda s: [x.strip() for x in s.split(';') if x.strip()])
    return df

class Preprocessor:
    def __init__(self, skills_vocab=None, interests_vocab=None):
        self.skills_mlb = MultiLabelBinarizer(classes=skills_vocab) if skills_vocab else MultiLabelBinarizer()
        self.interests_mlb = MultiLabelBinarizer(classes=interests_vocab) if interests_vocab else MultiLabelBinarizer()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.scaler = StandardScaler()

    def fit(self, df):
        df = split_skills_interests(df)
        self.skills_mlb.fit(df['skills_list'])
        self.interests_mlb.fit(df['interests_list'])
        cat_cols = df[['education_level','personality','work_preference']].fillna('NA')
        self.ohe.fit(cat_cols)
        num_cols = df[['aptitude_math','aptitude_verbal','aptitude_logical']].fillna(0)
        self.scaler.fit(num_cols)
        return self

    def transform(self, df):
        df = split_skills_interests(df)
        skills = self.skills_mlb.transform(df['skills_list'])
        interests = self.interests_mlb.transform(df['interests_list'])
        cat_cols = df[['education_level','personality','work_preference']].fillna('NA')
        cat = self.ohe.transform(cat_cols)
        num_cols = df[['aptitude_math','aptitude_verbal','aptitude_logical']].fillna(0)
        num = self.scaler.transform(num_cols)
        age = df[['age']].fillna(df['age'].mean()).to_numpy()
        X = np.hstack([age, num, cat, skills, interests])
        return X

    def save(self, path_prefix):
        joblib.dump(self, path_prefix + '_preprocessor.joblib')

    @staticmethod
    def load(path):
        return joblib.load(path)
