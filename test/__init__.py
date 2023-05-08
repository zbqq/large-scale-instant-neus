test = {}

def register(name):
    def decorator(cls):
        test[name] = cls
        return cls
    return decorator

