models={}
def register(name):
    def asd(model):
        models[name]=model
    return asd

def make(name,config):
    return models[name](config)

from . import tcnn_nerf