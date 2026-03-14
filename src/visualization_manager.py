import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class VisulizationManager:
    def __init__(self, history, y_test, y_pred):
        self.history =  history
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_loss_accuracy(self):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize= (8,6))

        #Loss Curve
        ax1.plot(self.history.history["loss"], label = "Training Loss", color = "#a02eb6")
        ax1.plot(self.history.history["val_loss"], label = "Validation Loss", color = "#7d367e" , linestyle = "--")
        ax1.set_title("Model Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        #Accuracy Curve
        ax2.plot(self.history.history["accuracy"], label= "Training Accuracy", color = "#1E402C")
        ax2.plot(self.history.history["val_accuracy"], label = "Validation Accuracy", color = '#27ae60', linestyle = "--" )
        ax2.set_title("Model Accuracy Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot= True, fmt = "d", cmap = "Greens", 
                    xticklabels=["Normal", "Fraud"],
                    yticklabels= ["Normal", "Fraud"])
        plt.title("Confusion Matrix")
        plt.ylabel("Real Value")
        plt.xlabel("Prediction Value")
        plt.show()
    
    def print_report(self):
        print("------Classification Report------")
        print(classification_report(self.y_test, self.y_pred))
        print("--------------------------")
    
    def show_final_output(self, amount, risk_score):
        print("\n-------------" )
        print(f"Transaction Amount: {amount}$")
        print(f"Risk Score: {risk_score:.2f}")
        if risk_score >= 0.5:
            print("Fraud Detected")
        else:
            print("Transaction Safe")
        print("---------------")
