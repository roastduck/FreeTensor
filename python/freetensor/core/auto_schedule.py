import freetensor_ffi as ffi
import xgboost as xgb
import numpy as np
import os


class AutoSchedule(ffi.AutoSchedule):

    def __init__(self,
                 schedule,
                 target,
                 device,
                 *,
                 population=64,
                 rand_ratio=0.1,
                 tag="",
                 min_block_size=0,
                 rule_set=None,
                 verbose=0):
        '''
        Automatic scheduler

        Parameters
        ----------
        schedule : Schedule
            A Schedule object to apply schedules onto
        target : Target
            The type of devices to compile to
        population : int
            How many programs to test in each iteration
        rand_ratio : float
            Portion of random programs in the population. Higher ratio focuses on
            exploration, while lower ratio focuses on exploitation
        rule_set : Optional[set]
            Explicitly control over what rules to use. None for defualt rules
        verbose : int
            Verbosity level. 0 = print nothing, 1 = print tuning progress, 2 = print
            extra info mation of each rule
        '''

        self.population = population
        self.n_random = int(population * rand_ratio)
        self.n_inherited = population - self.n_random
        self.model = None
        self.xgb_params = {}
        self.save_file_name = tag + "_xgb.model"
        if os.path.isfile(self.save_file_name):
            self.model = xgb.Booster()
            self.model.load_model(self.save_file_name)
        self.verbose = verbose

        def predict_func(features):
            return self.predict(features)

        def update_func(features, times):
            return self.update(features, times)

        super(AutoSchedule, self).__init__(schedule, target, device,
                                           predict_func, update_func, tag,
                                           min_block_size, rule_set, verbose)

    def set_params(self, *args, **kws):
        super(AutoSchedule, self).set_params(args, kws)

    def run(self, iteration):
        for i in range(iteration):
            if self.verbose >= 1:
                print("Iteration ", i)
            self.search_one_round(self.population, self.n_inherited,
                                  self.n_random)
        return self.get_best_schedule()

    def predict(self, features):
        if not self.model:
            return [1] * len(features)
        return self.model.predict(xgb.DMatrix(np.array(features), missing=-1))

    def update(self, features, times):
        dtrain = xgb.DMatrix(np.array(features), np.array(times), missing=-1)
        self.model = xgb.train(self.xgb_params, dtrain, xgb_model=self.model)
        self.model.save_model(self.save_file_name)
