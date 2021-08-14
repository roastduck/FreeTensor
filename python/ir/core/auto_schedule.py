import ffi
import xgboost as xgb
import numpy as np


class AutoSchedule(ffi.AutoSchedule):

    def set_params(self, *args, **kws):
        super(AutoSchedule, self).set_params(args, kws)

    def run(self, iteration):
        annotations, times = super(AutoSchedule, self).init(20)
        annotations = np.array(annotations)
        times = np.array(times)
        dtrain = xgb.DMatrix(annotations, times)
        params = {}
        booster = xgb.train(params, dtrain)
        block = iteration // 20
        for i in range(iteration):
            rand_num = max(0, 16 - i // block)
            print("iteration {} rand {}".format(i, rand_num))
            sketches = super(AutoSchedule, self).get_random_sketches(100)
            annotations = []
            for sketch in sketches:
                annotations.append(sketch.get_annotation())
            annotations = np.array(annotations)
            annotations = xgb.DMatrix(annotations)
            pred = booster.predict(annotations)
            sketch_preds = list(zip(pred, sketches))[rand_num:]
            sketch_preds.sort(key=lambda x: x[0])
            sketches[rand_num:] = list(zip(*sketch_preds))[1][:20 - rand_num]
            annotations, times = super(AutoSchedule,
                                       self).test_and_add(sketches)
            annotations = np.array(annotations)
            times = np.array(times)
            dtrain = xgb.DMatrix(annotations, times)
            booster = xgb.train(params, dtrain, xgb_model=booster)
        return super(AutoSchedule, self).get_best_schedule()
