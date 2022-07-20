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
                 explore_ratio=0.1,
                 tag="",
                 min_block_size=0,
                 continue_training=False,
                 random_seed=None,
                 rule_set=None,
                 verbose=0,
                 lower_func_submit_api=None):
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
        explore_ratio : float
            Portion of random programs in the population. Higher ratio focuses on
            exploration, while lower ratio focuses on exploitation
        continue_trianing : bool
            Continue to train an existing XGBoost model file if found
        random_seed : Optional[int]
            Random seed. Setting a deterministic random seed and using a fixed OpenMP
            thread count (since we are using thread-local random number generators)
            resulting deterministic pseudo random numbers, but please note that the
            whole auto-scheduling procedure is still non-deterministic, becuase it
            measures real performance. Default to a non-deterministic seed
        rule_set : Optional[set]
            Explicitly control over what rules to use. None for defualt rules
        verbose : int
            Verbosity level. 0 = print nothing, 1 = print tuning progress, 2 = print
            extra info mation of each rule
        '''

        self.population = population
        self.n_explore = int(population * explore_ratio)
        self.n_exploit = population - self.n_explore
        self.model = None
        self.xgb_params = {}
        self.save_file_name = tag + "_xgb.model"
        if continue_training and os.path.isfile(self.save_file_name):
            self.model = xgb.Booster()
            self.model.load_model(self.save_file_name)
        self.verbose = verbose
        self.lower_func_submit_api = lower_func_submit_api

        def predict_func(features):
            return self.predict(features)

        def update_func(features, times):
            return self.update(features, times)

        super(AutoSchedule,
              self).__init__(schedule, target, device, predict_func,
                             update_func, tag, min_block_size, random_seed,
                             rule_set, verbose, lower_func_submit_api)

    def set_params(self, *args, **kws):
        super(AutoSchedule, self).set_params(args, kws)

    def run(self, iteration):
        for i in range(iteration):
            if self.verbose >= 1:
                print("Iteration", i)
            self.search_one_round(self.population, self.n_exploit,
                                  self.n_explore)
        return self.get_best_schedule()

    def predict(self, features):
        if not self.model:
            return [1] * len(features)
        return self.model.predict(xgb.DMatrix(np.array(features), missing=-1))

    def update(self, features, times):
        dtrain = xgb.DMatrix(np.array(features), np.array(times), missing=-1)
        self.model = xgb.train(self.xgb_params, dtrain, xgb_model=self.model)
        self.model.save_model(self.save_file_name)
