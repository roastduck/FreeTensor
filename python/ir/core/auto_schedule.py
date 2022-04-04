import ffi
import xgboost as xgb
import numpy as np


class AutoSchedule(ffi.AutoSchedule):

    def __init__(self, schedule, target, device, n_measured):
        self.model = None
        self.xgb_params = {}

        def predict_func(features):
            return self.predict(features)

        def update_func(features, times):
            return self.update(features, times)

        super(AutoSchedule, self).__init__(schedule, target, device, n_measured,
                                           predict_func, update_func)

    def set_params(self, *args, **kws):
        super(AutoSchedule, self).set_params(args, kws)

    def run(self, iteration):
        for i in range(iteration):
            print("iter ", i)
            self.search_one_round(32)
        return self.get_best_schedule()

    def predict(self, features):
        if not self.model:
            return [1] * len(features)
        return self.model.predict(xgb.DMatrix(np.array(features), missing=-1))

    def update(self, features, times):
        dtrain = xgb.DMatrix(np.array(features), np.array(times), missing=-1)
        self.model = xgb.train(self.xgb_params, dtrain, xgb_model=self.model)
