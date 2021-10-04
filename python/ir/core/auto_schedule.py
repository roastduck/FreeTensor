import ffi
import xgboost as xgb
import numpy as np


class AutoSchedule:

    def __init__(self, schedule, target, device, nCandidates, nPredict) -> None:
        self.schedule = schedule
        self.target = target
        self.device = device
        self.nCandidates = nCandidates
        self.nPredict = nPredict
        self.sketches = []
        sketch = Sketch(self.schedule, self.target, self.device,
                        self.nCandidates, self.nPredict)
        self.sketches.append(sketch)
        rule_num = sketch.get_rule_num()
        for idx in range(rule_num):
            tmp_sketches = []
            for sketch_item in self.sketches:
                sketch_part_num = sketch_item.rule_analyse(idx)
                print("Found {} targets of rule {}.".format(
                    sketch_part_num, idx))
                tmp_tmp_sketches = [
                    Sketch(self.schedule, self.target, self.device,
                           self.nCandidates, self.nPredict)
                    for _ in range(sketch_part_num + 2)
                ]
                rule = sketch_item.get_rule(idx)
                current = sketch_item.get_current()
                base_sketch = sketch_item.get_base_sketch()
                for tmp_tmp_sketches_item in tmp_tmp_sketches:
                    tmp_tmp_sketches_item.set_rule(idx, rule)
                    tmp_tmp_sketches_item.set_current(current)
                    tmp_tmp_sketches_item.set_base_sketch(base_sketch)
                for tmp_tmp_sketches_idx in range(sketch_part_num):
                    print("Applying {}".format(tmp_tmp_sketches_idx))
                    tmp_tmp_sketches[tmp_tmp_sketches_idx].rule_apply(
                        idx, [tmp_tmp_sketches_idx])
                tmp_tmp_sketches[sketch_part_num].rule_apply(idx, [])
                tmp_tmp_sketches[sketch_part_num + 1].rule_apply(
                    idx, [x for x in range(sketch_part_num)])
                if sketch_part_num == 1:
                    tmp_tmp_sketches.pop()
                tmp_sketches.extend(tmp_tmp_sketches)
            self.sketches = tmp_sketches
        print("Generate {} sketches.".format(len(self.sketches)))

    def set_params(self, *args, **kws):
        for sketch in self.sketches:
            sketch.set_params(*args, **kws)

    def run(self, iteration):
        self.best_schedule = []
        for sketch in self.sketches:
            self.best_schedule.append(sketch.run(iteration))
        print("Got all sketches' final time.")
        best_time = self.best_schedule[0][1]
        best_idx = 0
        for idx in range(len(self.best_schedule)):
            if self.best_schedule[idx][1] < best_time:
                best_time = self.best_schedule[idx][1]
                best_idx = idx
        return self.best_schedule[best_idx][0]


class Sketch(ffi.AutoSchedule):

    def set_params(self, *args, **kws):
        super(Sketch, self).set_params(args, kws)

    def run(self, iteration):
        xgb_params = {}
        sketches = self.get_random_sketches(self.n_candidates())
        schedules = self.gen_schedules(sketches)
        features = self.gen_features(schedules)
        times = self.test_and_add(sketches, schedules)
        dtrain = xgb.DMatrix(np.array(features), np.array(times), missing=-1)
        booster = xgb.train(xgb_params, dtrain)
        for i in range(iteration):
            rand_num = int((iteration - i) * 0.8)
            print("iteration {}".format(i))
            sketches = self.get_random_sketches(self.n_predict())
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
        return (super(Sketch,
                      self).get_best_schedule(), super(Sketch,
                                                       self).get_best_time())
