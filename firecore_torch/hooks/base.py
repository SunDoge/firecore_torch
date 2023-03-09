

class BaseHook:

    def on_init(self, **kwargs):
        pass

    def before_epoch(self, **kwargs):
        pass

    def before_iter(self, **kwargs):
        pass

    def before_forward(self, **kwargs):
        pass

    def after_forward(self, **kwargs):
        pass

    def after_iter(self, **kwargs):
        pass

    def after_epoch(self, **kwargs):
        pass
