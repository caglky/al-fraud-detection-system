import matplotlib.pyplot as plt
import seaborn as sns

class EDAManager:
    def __init__ (self, df):
        self.df = df

    def show_analysis(self):
        print("----Exploratory Data Analysis Started---")
        plt.figure(figsize=(8,5))
        sns.countplot(x= "Class", data = self.df) 
        plt.title ("Fraud Distribution (0: Normal, 1: Fraud)")
        plt.xlabel ("Class")
        plt.ylabel("Process Number")
        plt.show()

        plt.figure(figsize=(8,5))
        sns.boxplot(x="Class", y="Amount", data=self.df)
        #plt.yscale("log")
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Class (0: Normal, 1: Fraud)")
        plt.ylabel("Amount")
        plt.show()

        plt.figure(figsize=(8,5))
        sns.histplot(self.df[self.df["Class"] == 0]["Time"], color="blue")
        plt.title("Time vs Process Density")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8,5))
        corr = self.df.corr() #correlation coefficient between -1 and 1
        sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
        #heatmap => cmap: color palette, annot= False: prevent the complexity if the data is so big
        plt.title("Correlation")
        plt.show()

        print("All EDA visual is showed")







