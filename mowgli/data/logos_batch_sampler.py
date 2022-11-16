

class LogosBatchSampler:
    """
    Base class for all batch samplers.
    """

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __state_dict(self):
        raise NotImplementedError

    def __load_state_dict(self):
        raise NotImplementedError
