# custom_exceptions.py
class BadClusteringError(ValueError):
    """Exception raised for errors in the clustering process."""
    def __init__(self, message="No appropriate clusters found. Adjust parameters."):
        self.message = message
        super().__init__(self.message)
