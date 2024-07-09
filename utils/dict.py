class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToObj(value)
            self.__dict__[key] = value

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        raise KeyError(f"'{key}'")

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)