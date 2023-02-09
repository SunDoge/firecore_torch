

class BaseHook:

    def before_epoch(self, **kwargs):
        pass

    def before_iter(self, **kwargs):
        pass

    def after_iter(self, **kwargs):
        pass

    def after_epoch(self, **kwargs):
        pass

    def after_metrics(self, **kwargs):
        pass
