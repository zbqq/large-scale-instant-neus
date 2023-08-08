test = {}
# 自底而上的注册逻辑
def register(name):
    def decorator(cls):
        test[name] = cls
        # return cls
    return decorator
# 自顶而下的分配逻辑
def make(name):
    return test[name]()
from . import test3
