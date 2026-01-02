import torch


class ActivationExtractor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.handles = []

        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._hook(name))
                self.handles.append(handle)

    def _hook(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu()
        return fn

    def clear(self):
        self.activations = {}

    def remove(self):
        for h in self.handles:
            h.remove()
