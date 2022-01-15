import ffi
import xgboost as xgb
import numpy as np


class AutoSchedule(ffi.AutoSchedule):

    def set_params(self, *args, **kws):
        super(AutoSchedule, self).set_params(args, kws)

    def run(self, iteration):
        xgb_params = {}
        sketches = self.search_one_round(self.n_candidates())
        schedules = self.gen_schedules(sketches)
        features = self.gen_features(schedules)
        times = self.test_and_add(sketches, schedules)
        dtrain = xgb.DMatrix(np.array(features), np.array(times), missing=-1)
        booster = xgb.train(xgb_params, dtrain)
        for i in range(iteration):
            rand_num = int((iteration - i) * 0.8)
            print("iteration {}".format(i))
            sketches = self.search_one_round(self.n_predict())
            schedules = self.gen_schedules(sketches)
            features = self.gen_features(schedules)
            pred = booster.predict(xgb.DMatrix(np.array(features), missing=-1))
            pack = list(zip(pred, sketches, schedules, features))
            pack[rand_num:].sort(key=lambda x: x[0])
            pack = pack[:self.n_candidates()]
            pred, sketches, schedules, features = zip(*pack)
            times = self.test_and_add(sketches, schedules)
            dtrain = xgb.DMatrix(np.array(features),
                                 np.array(times),
                                 missing=-1)
            booster = xgb.train(xgb_params, dtrain, xgb_model=booster)
        return super(AutoSchedule, self).get_best_schedule()
