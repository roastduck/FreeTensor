from __future__ import annotations
import queue
import threading
import time
import copy
from typing import Any, List
from typing import Dict
'''
task_type is specified
1.search_task
2.measure_task
(minus means result)
special param:
0.other(transfer control)
then check param
    trans_c:(those are conbined with additional param: time_stamp)
    0. logger
    1. broadcast availability
    2. return inavailability
    3. new_task_request    
minus means receive
'''


class Task(object):
    src_server_uid: str
    #
    target_server_uid: str
    #
    task_uid: int
    submit_uid: int
    #
    params: Any
    #
    task_type: int
    #
    block_start: int = 0
    block_end: int = 6400
    #
    normal_bundle_size: int = 4  # the number of tasks in a bundle
    min_bundle_size: int = 26  #the minimum size of a bundle(in blocks)
    single_task_block_size: int = 100  # the size of single task(in blocks)
    max_block_to_split: int = 1600  #no larger than the number of blocks will be further splitted

    #

    def split(self, _block_start: int, _block_end: int) -> Task:
        r = copy.copy(self)
        r.block_start = _block_start
        r.block_end = _block_end
        return r

    def run(self) -> None:
        print(self.block_start)
        time.sleep(1)
        print(self.block_end)

    def convert2dict(self) -> Dict:
        tmpdict: Dict = {
            "src_server_uid": self.src_server_uid,
            "target_server_uid": self.target_server_uid,
            "task_uid": self.task_uid,
            "submit_uid": self.submit_uid,
            "params": self.params,
            "task_type": self.task_type,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "normal_bundle_size": self.normal_bundle_size,
            "min_bundle_size": self.min_bundle_size,
            "single_task_block_size": self.single_task_block_size,
            "max_block_to_split": self.max_block_to_split
        }
        return tmpdict

    @classmethod
    def dict2self_base(cls, tmptask: Task, inputdict: Dict) -> Task:
        tmptask.src_server_uid = inputdict["src_server_uid"]
        tmptask.target_server_uid = inputdict["target_server_uid"]
        tmptask.task_uid = inputdict["task_uid"]
        tmptask.submit_uid = inputdict["submit_uid"]
        tmptask.params = inputdict["params"]
        tmptask.task_type = inputdict["task_type"]
        tmptask.block_start = inputdict["block_start"]
        tmptask.block_end = inputdict["block_end"]
        tmptask.normal_bundle_size = inputdict["normal_bundle_size"]
        tmptask.min_bundle_size = inputdict["min_bundle_size"]
        tmptask.single_task_block_size = inputdict["single_task_block_size"]
        tmptask.max_block_to_split = inputdict["max_block_to_split"]
        return tmptask

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Task:
        return cls.dict2self_base(Task(), inputdict)


class TaskRunner(threading.Thread):
    task: Task
    scheduler: RemoteTaskScheduler

    def __init__(self, _task: Task, _scheduler: RemoteTaskScheduler) -> None:
        self.task = _task
        self.scheduler = _scheduler
        super().__init__()

    def run(self):
        self.scheduler.execution_lock.acquire()
        return_val = self.task.run()
        self.scheduler.execution_lock.release()
        self.scheduler.execution_queue_cnt_lock.acquire()
        self.scheduler.execution_queue_cnt -= 1
        is_new_task_required = bool(self.scheduler.execution_queue_cnt < 5)
        self.scheduler.execution_queue_cnt_lock.release()
        self.scheduler.result_submit(self.task.src_server_uid, return_val)
        if is_new_task_required:
            self.scheduler.request_for_new_task_all()
        self.scheduler.server_list_lock.acquire()
        if len(self.scheduler.available_server_list) < 3:
            self.scheduler.request_for_new_task_all()
        self.scheduler.server_list_lock.release()


class TaskResult(object):
    src_server_uid: str
    #
    target_server_uid: str
    #
    task_uid: int
    submit_uid: int
    #
    task_type: int
    #
    block_start: int = 0
    block_end: int = 6400
    single_task_block_size: int = 100  # the size of single task(in blocks)
    #
    results: Any

    #
    @classmethod
    def get(cls, task: Task) -> TaskResult:
        return cls.get_base(task, TaskResult())

    @classmethod
    def get_base(cls, task: Task, tmp_taskresult :TaskResult) -> TaskResult:
        tmp_taskresult.src_server_uid = task.src_server_uid
        tmp_taskresult.target_server_uid = task.target_server_uid
        tmp_taskresult.task_uid = task.task_uid
        tmp_taskresult.submit_uid = task.submit_uid
        tmp_taskresult.task_type = task.task_type
        tmp_taskresult.block_start = task.block_start
        tmp_taskresult.block_end = task.block_end
        tmp_taskresult.single_task_block_size = task.single_task_block_size
        return tmp_taskresult

    def put_result(self, _results: Any):
        self.results = _results

    def merge(self, _task_result: TaskResult):
        pass

    def fetch(self) -> Any:
        return self.results

    def convert2dict(self) -> Dict:
        tmpdict: Dict = {
            "src_server_uid": self.src_server_uid,
            "target_server_uid": self.target_server_uid,
            "task_uid": self.task_uid,
            "submit_uid": self.submit_uid,
            "task_type": self.task_type,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "single_task_block_size": self.single_task_block_size,
            "results": self.results
        }
        return tmpdict

    @classmethod
    def dict2self_base(cls, tmptask_result: Any, inputdict: Dict) -> Any:
        tmptask_result.src_server_uid = inputdict["src_server_uid"]
        tmptask_result.target_server_uid = inputdict["target_server_uid"]
        tmptask_result.task_uid = inputdict["task_uid"]
        tmptask_result.submit_uid = inputdict["submit_uid"]
        tmptask_result.results = inputdict["results"]
        tmptask_result.task_type = inputdict["task_type"]
        tmptask_result.block_start = inputdict["block_start"]
        tmptask_result.block_end = inputdict["block_end"]
        tmptask_result.single_task_block_size = inputdict[
            "single_task_block_size"]
        return tmptask_result

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Task:
        return cls.dict2self_base(TaskResult(), inputdict)


class MeasureTask(Task):
    task_type: int = 2
    warmup_rounds: int
    attatched_params: tuple

    def __init__(self, _task_uid: int, run_times: int, _warmup_rounds: int,
                 _attatched_params: tuple, _params: List):
        self.task_uid = _task_uid
        self.single_task_block_size = run_times
        self.warmup_rounds = _warmup_rounds
        self.params = _params
        self.block_start = 0
        self.block_end = run_times * len(_params)
        self.attatched_params = _attatched_params

    def split(self, _block_start: int, _block_end: int) -> Task:
        tempparams = []
        for i in range(_block_start // self.single_task_block_size,
                       (_block_end + self.single_task_block_size - 1) //
                       self.single_task_block_size):
            tempparams.append(self.params[i])
        subtask = MeasureTask(self.task_uid, self.single_task_block_size,
                              self.warmup_rounds, self.attatched_params,
                              tempparams)
        subtask.block_start = _block_start
        subtask.block_end = _block_end
        return subtask

    def measure(self, rounds: int, warmups: int, attached_params: tuple,
                sketches: List) -> tuple[List[float], List[float]]:
        pass

    def run(self) -> TaskResult:
        tmptaskresult = MeasureResult.get(self)
        start = self.block_start
        end = self.block_end
        if (start >= end):
            return
        #first check if the first block is complete
        fbe: int = (start + self.single_task_block_size -
                    1) // self.single_task_block_size
        if (fbe * self.single_task_block_size) > end:
            tmpresult = self.measure(end - start, self.warmup_rounds,
                                     self.attached_params, self.params)
            tmptaskresult.put_result(tmpresult)
            return tmptaskresult
        blnow: int = 0
        if (fbe * self.single_task_block_size) > start:
            tmpresult = self.measure(fbe * self.single_task_block_size - start,
                                     self.warmup_rounds, self.attatched_params,
                                     self.params[0:1])
            blnow = 1
            start = fbe * self.single_task_block_size
        #for mid part
        blend = blnow + (end - start) // self.single_task_block_size
        if blend > blnow:
            tmpresult = tmpresult + self.measure(
                self.single_task_block_size, self.attatched_params,
                self.warmup_rounds, self.params[blnow, blend])
        start = (end //
                 self.single_task_block_size) * self.single_task_block_size
        #deal with the unfinished part
        if (start < end):
            tmpresult = tmpresult + self.measure(
                end - start, self.warmup_rounds, self.attached_params,
                self.params[blend, blend + 1])
        tmptaskresult.put_result(tmpresult)
        return tmptaskresult

    def convert2dict(self) -> Dict:
        tmpdict = super().convert2dict()
        tmpdict["warmup_rounds"] = self.warmup_rounds
        tmpdict["attatched_params"] = self.attatched_params
        return tmpdict

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Task:
        tmptask = super().dict2self_base(
            MeasureTask(0, 0, inputdict["warmup_rounds"],
                        inputdict["attatched_params"], []), inputdict)
        return tmptask


class MeasureResult(TaskResult):
    task_type: int = -2
    results: tuple[List[float], List[float]] = ([], [])

    #first element is avr, the second is stddev

    @classmethod
    def get(cls, task: Task) -> MeasureResult:
        return cls.get_base(task, MeasureResult())

    def merge(self, _measure_result: MeasureResult):
        start = _measure_result.block_start
        end = _measure_result.block_end
        if (start >= end):
            return
        #first check if the first block is complete
        fbe: int = (start + self.single_task_block_size -
                    1) // self.single_task_block_size
        if (fbe * self.single_task_block_size > end):
            self.results[0][fbe -
                            1] += (end - start) * _measure_result.results[0][0]
            self.results[1][fbe -
                            1] += (end - start) * (_measure_result.results[1][0]
                                                   **2)
            return
        blnow: int = 0
        if (fbe * self.single_task_block_size > start):
            self.results[0][fbe - 1] += (fbe * self.single_task_block_size -
                                         start) * _measure_result.results[0][0]
            self.results[1][fbe -
                            1] += (fbe * self.single_task_block_size -
                                   start) * (_measure_result.results[1][0]**2)
            blnow = 1
            start = fbe * self.single_task_block_size
        #for mid part
        while (start + self.single_task_block_size < end):
            self.results[0][fbe] += self.single_task_block_size * (
                _measure_result.results[0])[blnow]
            self.results[1][fbe] += self.single_task_block_size * (
                _measure_result.results[1][blnow]**2)
            blnow += 1
            fbe += 1
            start += self.single_task_block_size
        #deal with the unfinished part
        if (start < end):
            self.results[0][fbe] += (end - start) * (
                _measure_result.results[0])[blnow]
            self.results[1][fbe] += (end - start) * (
                _measure_result.results[1][blnow]**2)
        return

    def fetch(self) -> tuple[List[float], List[float]]:
        for i in range(len(self.results[0])):
            self.results[0][i] /= self.single_task_block_size
        for i in range(len(self.results[1])):
            self.results[1][i] /= self.single_task_block_size
            self.results[1][i] **= 0.5
        return self.results

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Task:
        return super().dict2self_base(MeasureResult(), inputdict)


class RemoteTaskScheduler(object):
    search_queue = queue.Queue()
    measure_queue = queue.Queue()
    execution_queue_cnt: int = 0

    submit_queue_lock = threading.Lock()
    execution_queue_cnt_lock = threading.Lock()
    execution_task_list = set()

    tasks_waiting_to_submit: Dict[int, Task] = {}
    #
    submit_lock_container: Dict[int, threading.Lock] = {
    }  #stores the general lock of a task(in case it won't return an incomplete value)
    merge_lock_container: Dict[int, threading.Lock] = {}  #stores merge_lock
    task_block_counter: Dict[int, int] = {}  #blocks left for a task
    task_result_container: Dict[int, TaskResult] = {}  #store task_result
    submitted_task_container: Dict[int, Task] = {}  #store submitted_task
    #
    task_uid_lock = threading.Lock()
    task_uid_global: int = 0
    submit_uid_lock = threading.Lock()
    submit_uid_global: int = 0
    #
    submit_lock_container_lock = threading.Lock()
    merge_lock_container_lock = threading.Lock()
    task_block_counter_lock = threading.Lock()
    task_result_container_lock = threading.Lock()
    submitted_task_container_lock = threading.Lock()
    tasks_waiting_to_submit_lock = threading.Lock()
    #
    execution_lock = threading.Lock()
    #
    server_list: Dict[str:tuple[int, float]] = {}
    available_server_list = set()
    #server_uid: tuple[server_status, last_connection_time]
    search_server_num: int = 0
    measure_server_num: int = 0
    server_list_lock = threading.Lock()
    #
    is_ready: bool = False
    self_server_uid: str

    #
    def __init__(self) -> None:
        self.submit_queue_lock.acquire()
        return

    def task_uid_assign(self) -> int:
        self.task_uid_lock.acquire()
        self.task_uid_global += 1
        tmp = self.task_uid_global
        self.task_uid_lock.release()
        return tmp

    def submit_uid_assign(self) -> int:
        self.submit_uid_lock.acquire()
        self.submit_uid_global += 1
        tmp = self.submit_uid_global
        self.submit_uid_lock.release()
        return tmp

    def task_auto_split(self, _task: Task) -> List[Task]:
        splitted_tasks: List[Task] = []
        tmp_start = _task.block_start
        tmp_end = _task.block_end
        normal_pace = _task.normal_bundle_size * _task.single_task_block_size
        while ((tmp_end - tmp_start) > _task.max_block_to_split):
            splitted_tasks.append(
                _task.split(tmp_start, tmp_start + normal_pace))
            tmp_start += normal_pace

        while (tmp_end - tmp_start) > 0:
            task_end = tmp_start + _task.single_task_block_size
            while (tmp_start + _task.min_bundle_size) < task_end:
                splitted_tasks.append(
                    _task.split(tmp_start, tmp_start + _task.min_bundle_size))
                tmp_start += _task.min_bundle_size

            if tmp_start < task_end:
                splitted_tasks.append(_task.split(tmp_start, task_end))
                tmp_start = task_end

        return splitted_tasks

    def task_register(self, _task: Task, _lock: threading.Lock) -> None:
        _lock.acquire()
        self.submit_lock_container_lock.acquire()
        self.submit_lock_container[_task.task_uid] = _lock
        self.submit_lock_container_lock.release()
        _task.src_server_uid = self.self_server_uid
        temp = self.task_auto_split(_task)
        self.submit_queue_lock.acquire()
        for subtask in temp:
            subtask.submit_uid = self.submit_uid_assign()
            self.tasks_waiting_to_submit_lock.acquire()
            self.tasks_waiting_to_submit[subtask.submit_uid] = subtask
            self.tasks_waiting_to_submit_lock.release()
            if subtask.task_type == 2:
                self.measure_queue.put(subtask.submit_uid)
            if subtask.task_type == 1:
                self.search_queue.put(subtask.submit_uid)
        self.submit_queue_lock.release()
        self.report_new_task()

    def task_merge(self, _task_result: TaskResult) -> None:
        #pop out the related submit_task
        self.submitted_task_container_lock.acquire()
        if not (_task_result.task_uid in self.submitted_task_container):
            self.submitted_task_container_lock.release()
            return
        self.submitted_task_container.pop(_task_result.task_uid)
        self.submitted_task_container_lock.release()
        #acquire the merge_lock
        self.merge_lock_container_lock.acquire()
        self.merge_lock_container[_task_result.task_uid].acquire()
        self.merge_lock_container_lock.release()
        #update the counter
        self.task_block_counter_lock.acquire()
        self.task_block_counter[
            _task_result.
            task_uid] -= _task_result.block_end - _task_result.block_start
        self.task_block_counter_lock.release()
        #merge_the_result
        self.task_result_container_lock.acquire()
        self.task_result_container[_task_result.task_uid].merge(_task_result)
        self.task_result_container_lock.release()
        #release the merge_lock
        self.merge_lock_container_lock.acquire()
        self.merge_lock_container[_task_result.task_uid].release()
        self.merge_lock_container_lock.release()
        #if all subtasks are done, remove the merge_lock, submit lock and free the memory
        if self.task_block_counter[_task_result.task_uid] <= 0:
            self.merge_lock_container.pop(_task_result.task_uid)
            self.task_block_counter.pop(_task_result.task_uid)
            self.submit_lock_container.pop(_task_result.task_uid).release()
        return

    def task_submit(self, _server_uid: str):
        task_availability: bool = False
        tmp_submit_uid: int
        tmp_task_type: int
        if self.server_list[_server_uid][0] == 1:
            self.submit_queue_lock.acquire()
            if self.search_queue.qsize() > 0:
                tmp_submit_uid = self.search_queue.get()
                task_availability = True
                tmp_task_type = 1
            self.submit_queue_lock.release()

        if self.server_list[_server_uid][0] == 2:
            self.submit_queue_lock.acquire()
            if self.measure_queue.qsize() > 0:
                tmp_submit_uid = self.measure_queue.get()
                task_availability = True
                tmp_task_type = 2
            self.submit_queue_lock.release()

        if self.server_list[_server_uid][0] == 3:
            self.submit_queue_lock.acquire()
            self.server_list_lock.acquire()
            if self.search_queue.qsize(
            ) * self.measure_server_num > self.measure_queue.qsize(
            ) * self.search_server_num:
                tmp_submit_uid = self.search_queue.get()
                task_availability = True
                tmp_task_type = 1
            else:
                if self.measure_queue.qsize() > 0:
                    tmp_submit_uid = self.measure_queue.get()
                    task_availability = True
                    tmp_task_type = 2
                else:
                    if self.search_queue.qsize() > 0:
                        tmp_submit_uid = self.search_queue.get()
                        task_availability = True
                        tmp_task_type = 1
            self.server_list_lock.release()
            self.submit_queue_lock.release()
        if task_availability:
            self.tasks_waiting_to_submit_lock.acquire()
            if self.send_tasks(
                    self.tasks_waiting_to_submit[tmp_submit_uid].convert2dict(),
                    _server_uid) == 0:
                self.submitted_task_container_lock.acquire()
                tmptask = self.tasks_waiting_to_submit.pop(tmp_submit_uid)
                tmptask.target_server_uid = _server_uid
                self.submitted_task_container[tmp_submit_uid] = tmptask
                self.submitted_task_container_lock.release
            else:
                if tmp_task_type == 1:
                    self.search_queue.put(tmp_submit_uid)
                if tmp_task_type == 2:
                    self.measure_queue.put(tmp_submit_uid)
            self.tasks_waiting_to_submit_lock.release()
        else:
            self.report_inavailability(_server_uid)

    def result_submit(self, _server_uid: int, _task_result: TaskResult):
        self.send_results(_task_result.convert2dict(), _server_uid)

    def task_trans_submit2waiting(self, submit_uid: int):
        self.submitted_task_container_lock.acquire()
        tmptask = self.submitted_task_container.pop(submit_uid)
        self.submitted_task_container_lock.release()
        self.tasks_waiting_to_submit_lock.acquire()
        self.tasks_waiting_to_submit[submit_uid] = tmptask
        self.tasks_waiting_to_submit_lock.release()
        if tmptask.task_type == 2:
            self.measure_queue.put(submit_uid)
        if tmptask.task_type == 1:
            self.search_queue.put(submit_uid)

    def report_inavailability(self, _server_uid: str):
        tmpdict: Dict = {
            "task_type": 0,
            "trans_c": 2,
            "time_stamp": time.time()
        }
        self.send_tasks(tmpdict, _server_uid)

    def report_new_task(self):
        tmpdict: Dict = {
            "task_type": 0,
            "trans_c": 1,
            "time_stamp": time.time()
        }
        task_type_list: List = []
        if self.search_queue.qsize() > 0:
            task_type_list.append(1)
        if self.measure_queue.qsize() > 0:
            task_type_list.append(2)
            if 1 in task_type_list:
                task_type_list.append(3)
        self.server_list_lock.acquire()
        for uidkey in self.server_list.keys():
            if self.server_list[uidkey][0] in task_type_list:
                t = threading.Thread(target=self.send_tasks,
                                     args=(tmpdict, uidkey))
                t.start()
        self.server_list_lock.release()

    def request_for_new_task(self, _server_uid: str):
        tmpdict: Dict = {
            "task_type": 0,
            "trans_c": 3,
            "time_stamp": time.time()
        }
        self.send_tasks(tmpdict, _server_uid)

    def request_for_new_task_all(self):
        for server_uid in self.available_server_list:
            threading.Thread(target=self.request_for_new_task,
                             args=(server_uid,))

    def send_tasks(self, _task: Dict, server_uid: str) -> int:
        pass
        #this part will use the method in RPCTools

    def send_results(self, _taskresult: Dict, server_uid: str) -> None:
        pass
        #this part will use the method in RPCTools

    def remote_measure_submit(
            self, rounds: int, warmups: int, attatched_params: int,
            Sketches: List[str]) -> tuple[List[float], List[float]]:
        tmpuid = self.task_uid_assign()
        tmptask = MeasureTask(tmpuid, rounds, warmups, attatched_params,
                              Sketches)
        tmplock = threading.Lock()
        self.task_register(tmptask, tmplock)
        tmplock.acquire()
        tmplock.release()
        self.task_result_container_lock.acquire()
        tmpresult = self.task_result_container.pop(tmpuid).fetch()
        self.task_result_container_lock.release()
        return tmpresult

    def remote_task_receive(self, src_host_uid: str, task: Dict) -> int:
        if task["task_type"] == 0:
            if task["trans_c"] == 0:
                pass
            elif task["trans_c"] == 1:
                self.update_availability(src_host_uid, task["time_stamp"])
            elif task["trans_c"] == 2:
                self.update_inavailability(src_host_uid, task["time_stamp"])
            elif task["trans_c"] == 3:
                threading.Thread(target=self.task_submit,
                                 args=(src_host_uid,))
        else:
            if task["task_type"] == 1:
                pass
            elif task["task_type"] == 2:
                tmptask = MeasureTask.dict2self(task)
            self.task_execution(tmptask)
        return 0

    def update_availability(self, server_uid: str, time_stamp: float):
        self.server_list_lock.acquire()
        if self.server_list[server_uid][1] < time_stamp:
            self.server_list[server_uid] = (self.server_list[server_uid][1],
                                            time_stamp)
            self.available_server_list.discard(server_uid)
        self.server_list_lock.release()
        if self.execution_queue_cnt < 5:
            threading.Thread(target=self.task_submit, args=(server_uid,))

    def update_inavailability(self, server_uid: str, time_stamp: float):
        self.server_list_lock.acquire()
        if self.server_list[server_uid][1] < time_stamp:
            self.server_list[server_uid] = (self.server_list[server_uid][1],
                                            time_stamp)
            self.available_server_list.add(server_uid)
        self.server_list_lock.release()

    def task_execution(self, task: Task):
        tmprunner = TaskRunner(task, self)
        self.execution_queue_cnt_lock.acquire()
        self.execution_queue_cnt += 1
        self.execution_queue_cnt_lock.release()
        tmprunner.start()

    def remote_result_receive(self, remote_host_uid: str,
                              task_result: TaskResult):
        if task_result["task_type"] == 1:
            pass
        elif task_result["task_type"] == 2:
            tmptask_result = MeasureResult.dict2self(task_result)
        self.task_merge(tmptask_result)

    def remote_logger(self, log: str):
        pass

    def remote_logger_execution(self, dict):
        pass

    def get_queue_len(self) -> Dict[int, int]:
        return {
            "0": self.execution_queue_cnt,
            "1": self.search_queue.qsize(),
            "2": self.measure_queue.qsize()
        }

    def add_host(self, _server_uid: str, sev_status: int) -> None:
        self.server_list_lock.acquire()
        self.server_list[_server_uid] = (sev_status, time.time())
        if sev_status == 1:
            self.search_server_num += 1
        if sev_status == 2:
            self.measure_server_num += 1
        if sev_status == 3:
            self.measure_server_num += 1
            self.search_server_num += 1
        if (self.measure_server_num + self.search_server_num) < 3:
            self.report_new_task()
        self.server_list_lock.release()

    def remove_host(self, server_uid: str) -> None:
        self.server_list_lock.acquire()
        sev_status = self.server_list.pop(server_uid)[0]
        if sev_status == 1:
            self.search_server_num -= 1
        if sev_status == 2:
            self.measure_server_num -= 1
        if sev_status == 3:
            self.measure_server_num -= 1
            self.search_server_num -= 1
        self.server_list_lock.release()
        self.submitted_task_container_lock.acquire()
        for key in self.submitted_task_container.keys():
            if self.submitted_task_container[
                    key].target_server_uid == server_uid:
                self.task_trans_submit2waiting(key)
        self.submitted_task_container_lock.release()

    def get_self_uid(self, server_uid: str) -> None:
        self.self_server_uid = server_uid
        if self.is_ready == False:
            self.submit_queue_lock.release()


#the followings are functional tests
'''
def remote_task_submit(remote_host_uid : int, task : Task) -> int:
    print(task.submit_uid)
    return 0

tmp: List[int] = []
for i in range(300):
    tmp.append(i)
task1 = MeasureTask(1,100,10,tmp)
task2 = task1.split(150,1000)
print(task2.params)
'''
