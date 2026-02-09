
def save_ml_model(ml_object, name):
    """
    Guarda modelos de ML
    """
    joblib.dump(ml_object, f"../models/{name}.joblib")
    print("Modelo guardado")


class MachineLearningProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.data = pd.read_csv("../data/heart.csv")
        self.pre_process_data()

    def pre_process_data(self):
        self.processed_data = self.data.drop(["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], axis = 1)
        
    def split_data(self):
        X = self.processed_data.drop("HeartDisease", axis = 1)
        y = self.processed_data["HeartDisease"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state = 100, test_size = 0.3, stratify= y)

    def train(self, model):
        print("1. Separando datos de train y test...")
        self.split_data()
        print(f"2. Entrando model {self.model_name}")
        self.fitted_model = model.fit(self.X_train, self.y_train)
        print("3. Entrenamiento finalizado")

    def predict(self):
        self.predictions = self.fitted_model.predict(self.X_test)
        print(classification_report(self.y_test, self.predictions))