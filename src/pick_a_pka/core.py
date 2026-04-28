# from .molgpka.model import predict as predict_molgpka  # Adjust to actual import
from .pkalearn.gnn.predict import predict as predict_pkalearn  # Adjust to actual import


class PKaPredictor:
    """
    Unified API to access multiple pKa prediction architectures.
    """

    def __init__(self, backend="molgpka", **kwargs):
        self.backend = backend.lower()
        if self.backend not in ["molgpka", "pkalearn"]:
            raise ValueError(f"Unknown backend: {self.backend}. Choose 'molgpka' or 'pkalearn'.")

        self.kwargs = kwargs

    def predict(self, input_data):
        # if self.backend == "molgpka":
        #     # Call MolGpKa's native logic
        #     return predict_molgpka(input_data, **self.kwargs)
        if self.backend == "pkalearn":
            # Call pKaLearn's native logic
            return predict_pkalearn(input_data, **self.kwargs)
