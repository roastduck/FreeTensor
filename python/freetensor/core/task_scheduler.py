from typing import List
import numpy as np

from .auto_schedule import AutoSchedule


class TaskScheduler:

    def __init__(self,
                 tasks: List[AutoSchedule],
                 *,
                 early_stopping=500,
                 measures_per_round=64,
                 backward_window=3,
                 alpha=0.2,
                 beta=2,
                 gamma=0.5):
        self.score_func = sum
        self.measures_per_round = measures_per_round
        self.backward_window = backward_window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tasks = tasks
        self.task_cts = [0] * len(tasks)
        self.task_best_cts = [0] * len(tasks)
        self.task_history = [[]] * len(tasks)
        self.bests = [1e30] * len(tasks)
        self.score = self.compute_score(self.bests)
        self.best_score = None
        self.dead_tasks = set()
        self.task_tags = []  # task_id -> tag
        self.tag_to_group_id = {}  # tag -> group_id
        self.group_task_ids = []  # group_id -> all task ids in this group
        self.flop_cts = []  # task_id -> the number of floating ops
        self.early_stopping = early_stopping
        for i, task in enumerate(self.tasks):
            tag = task.get_tag()
            self.task_tags.append(tag)
            self.flop_cts.append(task.get_flop())
            if not tag:
                continue

            if tag not in self.tag_to_group_id:
                self.tag_to_group_id[tag] = len(self.tag_to_group_id)
                self.group_task_ids.append([])
            self.group_task_ids[self.tag_to_group_id[tag]].append(i)

    def compute_score(self, costs):
        return self.score_func(costs)

    def tune_task(self, task_idx):
        print("tuning task ", task_idx)
        self.tasks[task_idx].search_one_round(self.measures_per_round)

        self.task_cts[task_idx] += 1

        best_time = self.tasks[task_idx].get_best_time()
        if best_time < self.bests[task_idx]:
            self.task_best_cts[task_idx] = self.task_cts[task_idx]
            self.bests[task_idx] = best_time

        no_change_trials = (
            self.task_cts[task_idx] -
            self.task_best_cts[task_idx]) * self.measures_per_round
        if no_change_trials > self.early_stopping:
            self.dead_tasks.add(task_idx)

        self.task_history[task_idx].append(self.bests[task_idx])

    def tune(self, rounds):
        print("start tuning")
        now_round = 0
        for i in range(len(self.tasks)):
            self.tune_task(i)
        self.best_score = self.compute_score(self.bests)
        while now_round < rounds and len(self.dead_tasks) < len(self.tasks):
            now_round += 1
            gradients = []
            for i in range(len(self.tasks)):
                if i in self.dead_tasks:
                    gradients.append(0)
                    continue

                # compute gradient from chain rule : (delta f / delta g_i)
                delta = 1e-4
                new_costs = list(self.bests)
                new_costs[i] -= delta
                chain_grad = (self.compute_score(self.bests) -
                              self.compute_score(new_costs)) / delta

                # compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
                if (self.task_cts[i] - 1 < len(self.task_history[i]) and
                        self.task_cts[i] - 1 - self.backward_window >= 0):
                    backward_grad = (
                        self.task_history[i][self.task_cts[i] - 1] -
                        self.task_history[i][self.task_cts[i] - 1 -
                                             self.backward_window]
                    ) / self.backward_window
                else:
                    backward_grad = 0

                # compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
                g_next_1 = self.bests[i] - (self.bests[i] / self.task_cts[i])

                g_next_2 = self.beta * 1e30
                group_id = self.tag_to_group_id.get(self.task_tags[i], None)
                if group_id is not None and len(
                        self.group_task_ids[group_id]) > 1:
                    best_flops = max([
                        self.flop_cts[j] / self.bests[j]
                        for j in self.group_task_ids[group_id]
                    ])
                    g_next_2 = self.beta * self.flop_cts[i] / best_flops

                g_next = min(g_next_1, g_next_2)
                forward_grad = g_next - self.bests[i]

                # combine all grads
                grad = chain_grad * (self.alpha * backward_grad +
                                     (1 - self.alpha) * forward_grad)
                assert grad <= 0
                gradients.append(grad)

            if max(gradients) == min(gradients):
                task_idx = np.random.choice(len(gradients))
            else:
                task_idx = np.argmin(gradients)

            self.tune_task(task_idx)
            cur_score = self.compute_score(self.bests)

            if cur_score < self.best_score:
                self.best_score = cur_score
