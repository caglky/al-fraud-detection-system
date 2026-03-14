from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

class PreprocessManager:
    def __init__ (self, df):
        self.df = df
        self.scaler = StandardScaler()
    
    def process_data(self):
        X= self.df.drop("Class", axis=1).values
        y= self.df["Class"].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state = 42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print("---Data Division Process Completed---")
        print("---Data is Normalized with StandartScaler---")
        print(f"--- Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}---")

        return X_train, X_val, X_test, y_train, y_val, y_test

