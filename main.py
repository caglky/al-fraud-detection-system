from src.data_loader import DataLoader
from src.eda_manager import EDAManager
from src.preprocess_manager import PreprocessManager
from src.model_manager import ModelManager
from src.visualization_manager import VisulizationManager


def main():
    loader = DataLoader("creditcard.csv")
    df = loader.load_data()

    eda = EDAManager(df)
    eda.show_analysis()

    preprocessor = PreprocessManager(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_data()
    
    model_unit = ModelManager(input_shape=X_train.shape[1])
    model_unit.train_model(X_train, y_train, X_test, y_test)

    history = model_unit.train_model(X_train, y_train, X_val, y_val)
    y_pred_probs = model_unit.model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")
    vis = VisulizationManager(history, y_test, y_pred)
    vis.plot_loss_accuracy()
    vis.plot_confusion_matrix()
    vis.print_report()

    sample_idx = 10
    sample_data = X_test[sample_idx].reshape(1,-1)
    risk_score = model_unit.model.predict(sample_data)[0][0]
    vis.show_final_output(320, risk_score)
    
    print("\n Project execution finished successfully!")

if __name__ == "__main__":
    main()

