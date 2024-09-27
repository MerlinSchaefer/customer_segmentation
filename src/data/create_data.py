import pandas as pd
from datetime import datetime
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

class DataSynthesizer:
    def __init__(self, filename):
        self.metadata = SingleTableMetadata()
        self.customer_data = self.load_data(filename)

    def load_data(self, filename) -> pd.DataFrame:
        """
        Load data from a csv file.
        """
        return pd.read_csv(filename)

    def detect_metadata(self):
        """
        Detect metadata from the loaded data.
        """
        self.metadata.detect_from_dataframe(self.customer_data)

    def create_synthesizer(self) -> TVAESynthesizer:
        """
        Create a TVAESynthesizer instance with the detected metadata.
        """
        return TVAESynthesizer(self.metadata)

    def create_test_data(self, num_rows: int = 500) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create a synthetic test data set of num_rows rows.
        """
        synthesizer = self.create_synthesizer()
        synthesizer.fit(self.customer_data)
        synthetic_data = synthesizer.sample(num_rows=num_rows)
        synthetic_target = synthetic_data["Segmentation"]
        synthetic_features = synthetic_data.drop("Segmentation", axis=1)
        return synthetic_features, synthetic_target

    def create_train_data(self, num_rows: int = 500) -> pd.DataFrame:
        """
        Create a synthetic train data set of num_rows rows.
        """
        synthesizer = self.create_synthesizer()
        synthesizer.fit(self.customer_data)
        return synthesizer.sample(num_rows=num_rows)

if __name__ == "__main__":
    today = datetime.datetime.today.strftime("%Y-%m-%d")

    synthesizer = DataSynthesizer("./original/Train.csv")
    synthesizer.detect_metadata()
    test_features, test_target = synthesizer.create_test_data()
    train_data = synthesizer.create_train_data()

    # TODO: create pipeline to generate and load data to cloud storage
    test_features.to_csv(f"./synthetic/{today}_test_features.csv")
    test_target.to_csv(f"./synthetic/{today}_test_target.csv")
    train_data.to_csv(f"./synthetic/{today}_train.csv")

