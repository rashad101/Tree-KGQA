class DatasetNotFoundException(Exception):
    """Exception raised for unknown datasets.

    Attributes:
        dataset -- name of the dataset
    """

    def __init__(self, dataset):
        self.message = f"Dataset:{dataset} not found !! Valid datasets: webqsp, lcquad-2.0"
        super().__init__(self.message)


class UnknownTaskException(Exception):
    """Exception raised for unknown tasks.

    Attributes:
        task -- name of the task
    """

    def __init__(self, task):
        self.task = task
        self.message = f"Task:{task} is not recognized !! Valid tasks: EL, KGQA"
        super().__init__(self.message)