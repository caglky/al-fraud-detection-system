import pandas as pd 
import os 

class DataLoader:
    def __init__(self, filename):
        self.path = os.path.join(os.path.dirname(__file__),"..","data", filename)
        self.df = None
    def load_data (self): 
        try:
            self.df = pd.read_csv(self.path)

            print("----Data First Report----")
            print("First 5 Rows")
            print(self.df.head())
            print("Data Info")
            print(self.df.info())
            print("Data Describe")
            print(self.df.describe)

            rows, cols = self.df.shape
            features = cols - 1 # ı deleted the class column
            if "Class" in self.df.columns:
                fraud_count = self.df["Class"].sum()
                fraud_rate = (fraud_count / rows) * 100
            else:
                fraud_rate = 0
                fraud_count = "Not found"
            print("-----------")
            print(f"Data Dimension: {rows} rows")
            print(f"Feature Number: {features}")
            print(f"Fraud Number: {fraud_count}")
            print(f"Fraud Rate: %{fraud_rate:.4f}")

            return self.df
        except Exception as e:
            print(f"Error: {e}")
            return None
        

    