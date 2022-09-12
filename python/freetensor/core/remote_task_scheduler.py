from __future__ import annotations
import queue
import threading
import time
import freetensor_ffi as ffi
from typing import Any, List
from typing import Dict
from .. import core
'''
task_type is specified
search_task  ("search")
measure_task  ("measure")
special param:
"control"
then check param
    trans_c:(those are conbined with additional param: time_stamp)
    log. logger
    avail. broadcast availability
    inavail. return inavailability
    task_request. new_task_request
    q_check. check whether a task is in the execution queue
'''
'''
How does this part work:
1. commit a task via methods like RemoteTaskScheduler.remote_measure_submit()
    in this part, the task will be transformed into a Task object, then:
        1.1 a unique threading lock is acquired
        1.2 the task is pushed into one of the submit queue(measure queue, search queue)
        1.3 a message is broadcasted to all workers, telling them that new tasks are available to fetch

2. a worker receive the broadcast in 1.3, and update the server's availability
    if needed, the worker will send requests to all
    available servers for new tasks (currently, it
    means the number tasks in execution queue < 5)

3. the client receive the request in 2, and send tasks to the worker(one for a time),
    if no tasks are available, it will send
    another message, telling that the server's
    submit queues are not available

3. when a worker receives a task, it will check whether it is a task or a control message
    if it is a task, the task will be used to construct a TaskRunner object to run separately.
    threadlock is also acquired to make sure that only one task is running at a time
    The return value is a TaskResult object.the object is then transformed into a dict and sent back to the source

4. When a TaskResult object is received by client, it immediately checks whether it is valid.
    If is, the result will be stored.
    The lock in 1.1 will be released.

5. then the method in 1 returns the return value

'''


class Task(object):  #the parent class of all tasks
    src_server_uid: str = ""
    #
    target_server_uid: str = ""
    #
    task_uid: int = 0
    submit_uid: int = 0
    #
    params: Any  #the params of the task
    #
    task_type: str = ""  #the type of the task (search, measure, etc.)
    host_type: str = ""  #the type of host you want to run the task on

    #

    def run(self) -> None:
        #this part runs the task in runner(remote)
        print("running")
        time.sleep(1)

    def convert2dict(self) -> Dict:
        #this part converts the task to dictionary(so that it can be transferred via network)
        return self.__dict__

    @classmethod
    def dict2self_base(cls, tmptask: Task, inputdict: Dict) -> Task:
        #the base of the rev_progress of convert2dict
        for key, value in inputdict.items():
            setattr(tmptask, key, value)
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
        self.scheduler.append_to_execution_queue(self.task.src_server_uid,
                                                 self.task.submit_uid)
        self.scheduler.execution_lock.acquire()
        #acquire lock so that only one task can run at a time
        return_val = self.task.run()
        #run the task
        self.scheduler.execution_lock.release()
        #release the lock

        self.scheduler.execution_queue_cnt_lock.acquire()
        self.scheduler.execution_queue_cnt -= 1
        #maintain the execution_queue
        #this means that the task is now waiting to be removed from the queue
        is_new_task_required = bool(self.scheduler.execution_queue_cnt < 5)
        #check whether a new task is required
        self.scheduler.execution_queue_cnt_lock.release()

        self.scheduler.result_submit(self.task.src_server_uid, return_val)
        #send the result back to src

        if self.scheduler.verbose > 0:
            print(self.scheduler.get_queue_len())
        if is_new_task_required:
            for i in range(2):
                self.scheduler.request_for_new_task(self.task.src_server_uid)
            self.scheduler.request_for_new_task_all()
        # fetch tasks if there are too few tasks to execute
        time.sleep(3)
        self.scheduler.remove_from_execution_queue(self.task.src_server_uid,
                                                   self.task.submit_uid)
        # remove the task from execution queue completely


class TaskResult(object):
    #the meaning of the following params are the same as those in Task
    src_server_uid: str
    #
    target_server_uid: str
    #
    task_uid: int
    submit_uid: int
    #
    task_type: str
    #
    results: Any  #the result of the task

    #the following two get the required params from the task
    @classmethod
    def get(cls, task: Task) -> TaskResult:
        return cls.get_base(task, TaskResult())

    @classmethod
    def get_base(cls, task: Task, tmp_taskresult: TaskResult) -> TaskResult:
        tmp_taskresult.src_server_uid = task.src_server_uid
        tmp_taskresult.target_server_uid = task.target_server_uid
        tmp_taskresult.task_uid = task.task_uid
        tmp_taskresult.submit_uid = task.submit_uid
        tmp_taskresult.task_type = task.task_type
        return tmp_taskresult

    def put_result(self, _results: Any):
        self.results = _results

    def fetch(self) -> Any:
        #get the merged result from TaskResult object
        return self.results

    def convert2dict(self) -> Dict:
        #this part converts the class to dictionary(so that it can be transferred via network)
        return self.__dict__

    @classmethod
    def dict2self_base(cls, tmptask_result: Any, inputdict: Dict) -> Any:
        #the base of the rev_progress of convert2dict
        for key, value in inputdict.items():
            setattr(tmptask_result, key, value)
        return tmptask_result

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Task:
        return cls.dict2self_base(TaskResult(), inputdict)


class MeasureTask(Task):
    warmup_rounds: int
    run_times: int
    attached_params: tuple

    #a block means one round of timing rounds

    def __init__(self, _task_uid: int, _run_times: int, _warmup_rounds: int,
                 _attached_params: tuple, _params: List):
        self.task_type = "measure"
        self.task_uid = _task_uid
        self.warmup_rounds = _warmup_rounds
        self.params = _params
        self.attached_params = _attached_params
        self.run_times = _run_times

    @classmethod
    def args_deserialization(cls, attached_params: tuple,
                             params: List) -> tuple:
        target, device, args, kws = attached_params
        if type(target[1]) is not bytes:
            target[1] = target[1].data
        if type(device[1]) is not bytes:
            device[1] = device[1].data
        for array in args:
            if type(array[1]) is not bytes:
                array[1] = array[1].data
        for key, array in kws.items():
            if type(array[1]) is not bytes:
                array[1] = array[1].data

        return ((ffi.load_target(target), ffi.load_device(device),
                 [ffi.load_array(array) for array in args
                 ], {key: ffi.load_array(array) for key, array in kws.items()}),
                [ffi.load_ast(func) for func in params])

    @classmethod
    def args_serialization(cls, attached_params: tuple, params: List) -> tuple:
        target, device, args, kws = attached_params
        return ((ffi.dump_target(target), ffi.dump_device(device),
                 [ffi.dump_array(array) for array in args
                 ], {key: ffi.dump_array(array) for key, array in kws.items()}),
                [ffi.dump_ast(func) for func in params])

    def measure(self, rounds: int, warmups: int, attached_params: tuple,
                sketches: List) -> tuple[List[float], List[float]]:
        return ffi.rpc_measure(
            rounds, warmups,
            *(MeasureTask.args_deserialization(attached_params, sketches)))
        #this part will use the measure method of cpp-python-bridge in remote machine
        #the method is separated for easier maintenance

    def run(self) -> TaskResult:
        #run the measure progress and get the result
        tmpresult = MeasureResult.get(self)
        tmp = self.measure(self.run_times, self.warmup_rounds,
                           self.attached_params, self.params)
        tmpresult.put_result(tmp)
        return tmpresult

    def convert2dict(self) -> Dict:
        #this part converts the class to dictionary(so that it can be transferred via network)
        return self.__dict__

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Task:
        #the rev_progress of convert2dict
        tmptask = super().dict2self_base(
            MeasureTask(0, 0, inputdict["warmup_rounds"],
                        inputdict["attached_params"], []), inputdict)
        return tmptask


class MeasureResult(TaskResult):
    task_type: str = "measure"
    results: tuple[List[float], List[float]] = ([], [])

    #first element is avr, the second is stddev

    @classmethod
    def get(cls, task: Task) -> MeasureResult:
        return cls.get_base(task, MeasureResult())

    def fetch(self) -> tuple[List[float], List[float]]:
        return self.results

    @classmethod
    def dict2self(cls, inputdict: Dict) -> TaskResult:
        return super().dict2self_base(MeasureResult(), inputdict)


class RemoteTaskScheduler(object):
    rpctool: core.RPCTools.RPCTool  #the binded rpctool

    task_queues: Dict[str, queue.Queue] = {}  #host_type: the queueu
    execution_queue_cnt: int = 0  #tell how many tasks are to run

    execution_queue_cnt_lock = threading.Lock(
    )  #the lock to make sure that execution_queue_cnt don't get wrong

    tasks_waiting_to_submit: Dict[int, Task] = {}
    #
    submit_lock_container: Dict[int, threading.Event] = {
    }  #stores the general lock of a task(in case it won't return an incomplete value)
    task_result_container: Dict[int, TaskResult] = {}  #store task_result
    submitted_task_container: Dict[int, Task] = {}  #store submitted_task
    #
    task_uid_lock = threading.Lock()
    task_uid_global: int = 0
    submit_uid_lock = threading.Lock()
    submit_uid_global: int = 0
    #
    submitted_task_container_lock = threading.Lock()
    #
    execution_lock = threading.Lock(
    )  #the execution_lock makes sure that only one task can be executed at a time
    #
    server_list: Dict[str:tuple[List[str], float]] = {}
    #server_uid: tuple[server_status, last_connection_time]
    available_server_list: Dict[str:int] = {}
    #(available_server_uid: tries)

    servernums: Dict[str, int] = {}
    #the number of different host types, host_type: host_numbers
    server_list_lock = threading.Lock()
    #initialize from remote

    self_server_uid: str
    self_sev_status: List

    execution_tasks: Dict[str:set] = {}
    #task in execution_queue, host_uid: tasks
    execution_tasks_check_lock = threading.Lock()

    #the following are for diagnose use
    recalls: int = 0
    verbose: int = 1
    inavailability_counter_lock = threading.Lock()
    inavailability_counter: int = 0

    init_lock: bool
    global_tries: int = 5

    #

    def __init__(self, sev_status=["default"]) -> None:
        self.init_lock = False
        self.self_server_uid = "localhost"
        self.add_host("localhost", sev_status)
        #put self in server_list, so that it can run locally
        self.self_sev_status = sev_status
        self.verbose = 0
        return

    def bind_rpctool(self, _rpctool: core.RPCTools.RPCTool) -> None:
        self.rpctool = _rpctool

    def task_uid_assign(self) -> int:
        #assign a task_uid
        self.task_uid_lock.acquire()
        self.task_uid_global += 1
        tmp = self.task_uid_global
        self.task_uid_lock.release()
        return tmp

    def submit_uid_assign(self) -> int:
        #assign a submit_uid
        self.submit_uid_lock.acquire()
        self.submit_uid_global += 1
        tmp = self.submit_uid_global
        self.submit_uid_lock.release()
        return tmp

    def task_register(self, _task: Task, _lock: threading.Event) -> None:
        _lock.clear()  #acquire the threadlock

        self.submit_lock_container[_task.task_uid] = _lock
        #put threadlock in submit_lock_container
        _task.src_server_uid = self.self_server_uid
        #write src_server_uid to the task (if not initialized online, it will be localhost)
        _task.submit_uid = self.submit_uid_assign()
        #get submit_uid
        self.tasks_waiting_to_submit[_task.submit_uid] = _task
        #put the task to tasks_waiting_to_submit
        self.task_queues.setdefault(_task.host_type, queue.Queue())
        #ensure that a queue exists(or a new queue will be created)
        self.task_queues[_task.host_type].put(_task.submit_uid)
        #put these subtasks in submit queues
        self.report_new_task()
        #report that tasks are available to fetch

    def result_process(self, _task_result: TaskResult) -> None:
        #pop out the related submit_task and prevent double receive
        self.submitted_task_container_lock.acquire()
        if not (_task_result.submit_uid in self.submitted_task_container):
            self.submitted_task_container_lock.release()
            return
        self.submitted_task_container.pop(_task_result.submit_uid)
        self.submitted_task_container_lock.release()

        #put the result to container
        self.task_result_container[_task_result.task_uid] = _task_result

        #unblock the task and free the memory
        self.submit_lock_container.pop(_task_result.task_uid).set()
        return

    def task_submit(self, _server_uid: str):
        #automatically submit a task and balance the queue
        if self.verbose > 0:
            print("submitting tasks to server: " + _server_uid)
        task_availability: bool = False
        tmp_submit_uid: int

        #get a submit_uid from a non-empty queue
        avail_host_type = self.server_list[_server_uid][0]
        for i in avail_host_type:
            if i in self.task_queues:
                try:
                    tmp_submit_uid = self.task_queues[i].get(block=False)
                except queue.Empty:
                    pass
                else:
                    task_availability = True
                    break
        #send the task to remote host
        if task_availability:
            tmptask = self.tasks_waiting_to_submit.pop(tmp_submit_uid)
            self.submitted_task_container[tmp_submit_uid] = tmptask
            if self.send_tasks(tmptask.convert2dict(), _server_uid) == 0:
                tmptask.target_server_uid = _server_uid
                tmpthread = threading.Thread(
                    target=self.remote_check_in_execution_queue,
                    args=(tmptask.src_server_uid, _server_uid,
                          tmptask.submit_uid))
                tmpthread.start()
                #if task is successfully sent, modify the target_server_uid
            else:
                self.task_trans_submit2waiting(tmp_submit_uid)
                #if not, put the task back to the end of the queue

        else:
            self.report_inavailability(_server_uid)
            #if no tasks are available, report inavailability

    def remote_check_in_execution_queue(self, src_server_uid: str,
                                        target_server_uid: str,
                                        submit_uid: int) -> None:
        tmpdict: Dict = {
            "task_type": "control",
            "trans_c": "q_check",
            "src_server_uid": src_server_uid,
            "submit_uid": submit_uid
        }
        time.sleep(3)
        while (submit_uid in self.submitted_task_container):
            if self.send_tasks(tmpdict, target_server_uid) == 0:
                pass
            else:
                self.task_trans_submit2waiting(submit_uid)
                self.report_new_task()
                return
            time.sleep(0.5)

    def result_submit(self, _server_uid: int, _task_result: TaskResult):
        self.send_results(_task_result.convert2dict(), _server_uid)

    def task_trans_submit2waiting(self, submit_uid: int):
        #put task from submitted_task_container to tasks_waiting_to_submit
        self.recalls += 1
        #recall counter
        self.submitted_task_container_lock.acquire()
        if not (submit_uid in self.submitted_task_container):
            #prevent double recall
            self.submitted_task_container_lock.release()
            return
        tmptask = self.submitted_task_container.pop(submit_uid)
        self.submitted_task_container_lock.release()
        submit_uid = self.submit_uid_assign()
        tmptask.submit_uid = submit_uid
        self.tasks_waiting_to_submit[submit_uid] = tmptask
        self.task_queues[tmptask.host_type].put(submit_uid)

    def report_inavailability(self, _server_uid: str):
        tmpdict: Dict = {
            "task_type": "control",
            "trans_c": "inavail",
            "time_stamp": time.time()
        }
        self.inavailability_counter_lock.acquire()
        self.inavailability_counter += 1
        self.inavailability_counter_lock.release()
        if self.verbose > 0:
            print("reporting inavailability")
        self.send_tasks(tmpdict, _server_uid)

    def report_availability(self, _server_uid: str, tmpdict: Dict):
        tries = self.global_tries
        while (tries > 0):
            tries -= 1
            if self.send_tasks(tmpdict, _server_uid) == 0:
                return
            time.sleep(0.5)

    def report_new_task(self):
        #broadcast the information that new tasks are available to fetch
        if self.verbose > 0:
            print("reporting availability")
        tmpdict: Dict = {
            "task_type": "control",
            "trans_c": "avail",
            "time_stamp": time.time()
        }
        #check which host type is available
        host_type_list = set()
        for host_type in self.task_queues.keys():
            if self.task_queues[host_type].qsize() > 0:
                host_type_list.add(host_type)

        #send the task to right servers in order to prevent unnecessary overhead
        self.server_list_lock.acquire()
        for uidkey in self.server_list.keys():
            tmpbool = False
            for i in self.server_list[uidkey][0]:
                if i in host_type_list:
                    tmpbool = True
                    break
            if tmpbool:
                t = threading.Thread(target=self.report_availability,
                                     args=(uidkey, tmpdict))
                t.start()
        self.server_list_lock.release()
        if self.verbose > 0:
            print("availability reported")

    def request_for_new_task(self, _server_uid: str):
        tmpdict: Dict = {
            "task_type": "control",
            "trans_c": "task_request",
            "time_stamp": time.time()
        }
        tries = self.global_tries
        while tries > 0:
            tries -= 1
            if self.send_tasks(tmpdict, _server_uid) == 0:
                return
            time.sleep(0.5)

    def request_for_new_task_all(self):
        if self.verbose > 0:
            print("requesting_tasks")
        tmplist = self.available_server_list.keys()
        for server_uid in tmplist:
            t = threading.Thread(target=self.request_for_new_task,
                                 args=(server_uid,))
            t.start()

    def is_in_execution_queue(self, src_server_uid: str,
                              submit_uid: int) -> int:
        self.execution_tasks_check_lock.acquire()
        retval = -1
        if not (src_server_uid in self.execution_tasks):
            self.execution_tasks_check_lock.release()
            return -1
        else:
            tmpset = self.execution_tasks[src_server_uid]
            if submit_uid in tmpset:
                retval = 0
        self.execution_tasks_check_lock.release()
        return retval

    def append_to_execution_queue(self, src_server_uid: str,
                                  submit_uid: int) -> None:
        self.execution_tasks_check_lock.acquire()
        self.execution_tasks.setdefault(src_server_uid, set())
        self.execution_tasks[src_server_uid].add(submit_uid)
        self.execution_tasks_check_lock.release()

    def remove_from_execution_queue(self, src_server_uid: str,
                                    submit_uid: int) -> None:
        self.execution_tasks_check_lock.acquire()
        if not (src_server_uid in self.execution_tasks):
            return
        self.execution_tasks[src_server_uid].discard(submit_uid)
        if len(self.execution_tasks[src_server_uid]) == 0:
            self.execution_tasks.pop(src_server_uid)
        self.execution_tasks_check_lock.release()

    def send_tasks(self, _task: Dict, server_uid: str) -> int:
        _task.setdefault("time_stamp", time.time())
        if self.verbose > 0:
            print("sending tasks to " + server_uid)
        if (server_uid == "localhost") or (server_uid == self.self_server_uid):
            return self.remote_task_receive(self.self_server_uid, _task)
        else:
            if self.init_lock:
                return self.rpctool.remote_task_submit(server_uid, _task)
            else:
                return -1
        #this part will use the method in RPCTools

    def send_results(self, _taskresult: Dict, server_uid: str) -> None:
        if self.verbose > 0:
            print("sending results to" + server_uid)
        if (server_uid == "localhost") or (server_uid == self.self_server_uid):
            self.remote_result_receive(self.self_server_uid, _taskresult)
            return
        else:
            if self.init_lock:
                self.rpctool.remote_result_submit(server_uid, _taskresult)

        #this part will use the method in RPCTools

    def remote_measure_submit(
            self,
            rounds: int,
            warmups: int,
            attached_params: int,
            Sketches: List[str],
            host_type="default") -> tuple[List[float], List[float]]:
        #the method used to submit measure task by cpp-python-bridge
        tmpuid = self.task_uid_assign()
        tmptask = MeasureTask(
            tmpuid, rounds, warmups,
            *(MeasureTask.args_serialization(attached_params, Sketches)))
        tmptask.host_type = host_type
        tmplock = threading.Event()
        self.task_register(tmptask, tmplock)
        tmplock.wait()
        tmpresult = self.task_result_container.pop(tmpuid).fetch()
        return tmpresult

    def remote_task_receive(self, src_host_uid: str, task: Dict) -> int:
        #receive and decoding(refer to the beginning of the file)
        if task["task_type"] == "control":
            if task["trans_c"] == "log":
                pass
            elif task["trans_c"] == "avail":
                self.update_availability(src_host_uid, task["time_stamp"])
            elif task["trans_c"] == "inavail":
                self.update_inavailability(src_host_uid, task["time_stamp"])
            elif task["trans_c"] == "task_request":
                t = threading.Thread(target=self.task_submit,
                                     args=(src_host_uid,))
                t.start()
            elif task["trans_c"] == "q_check":
                return self.is_in_execution_queue(task["src_server_uid"],
                                                  task["submit_uid"])
        else:
            if self.verbose > 0:
                print("receiving")
            self.update_availability(src_host_uid, task["time_stamp"])
            if task["task_type"] == "search":
                pass
            elif task["task_type"] == "measure":
                tmptask = MeasureTask.dict2self(task)

            self.task_execution(tmptask)
        return 0

    def update_availability(self, server_uid: str, time_stamp: float):
        #update the availability of a specified server
        self.server_list_lock.acquire()
        if not server_uid in self.server_list:
            self.server_list_lock.release()
            return
        if self.server_list[server_uid][1] < time_stamp:
            self.server_list[server_uid] = (self.server_list[server_uid][0],
                                            time_stamp)
            self.available_server_list[server_uid] = 10
        self.server_list_lock.release()
        tmp = self.execution_queue_cnt
        if tmp == 0:
            for i in range(3):
                t = threading.Thread(target=self.request_for_new_task,
                                     args=(server_uid,))
                t.start()

    def update_inavailability(self, server_uid: str, time_stamp: float):
        #remove the server from the available server list
        self.server_list_lock.acquire()
        if self.server_list[server_uid][1] < (time_stamp - 1):
            self.server_list[server_uid] = (self.server_list[server_uid][0],
                                            time_stamp)
            if server_uid in self.available_server_list:
                self.available_server_list[server_uid] -= 1
                if self.available_server_list[server_uid] < 0:
                    self.available_server_list.pop(server_uid)
                    t = threading.Thread(target=self.rebuild_task_connection,
                                         args=(server_uid,))
                    t.start()
        self.server_list_lock.release()

    def rebuild_task_connection(self, server_uid: str):
        time.sleep(2)
        tries = self.global_tries
        while tries > 0:
            tries -= 1
            if server_uid in self.available_server_list:
                return
            self.request_for_new_task(server_uid)
            time.sleep(0.5)

    def task_execution(self, task: Task):  #execute the task(from remote_server)
        tmprunner = TaskRunner(task, self)
        if self.verbose > 0:
            print(self.get_queue_len())
        self.execution_queue_cnt_lock.acquire()
        self.execution_queue_cnt += 1
        self.execution_queue_cnt_lock.release()
        tmprunner.start()

    def remote_result_receive(self, remote_host_uid: str,
                              task_result: TaskResult):
        #get results from remote_server
        if task_result["task_type"] == "search":
            pass
        elif task_result["task_type"] == "measure":
            tmptask_result = MeasureResult.dict2self(task_result)
            #self.task_merge(tmptask_result)
            t = threading.Thread(target=self.result_process,
                                 args=(tmptask_result,))
            t.start()

    def remote_logger(self, log: str):
        pass

    def remote_logger_execution(self, dict):
        pass

    def get_queue_len(self) -> Dict[int, int]:
        tmpdict = {"execution_queue": self.execution_queue_cnt}
        for i in self.task_queues.keys():
            tmpdict[i] = self.task_queues[i].qsize()
        return tmpdict

    def add_host(self, _server_uid: str, sev_status: List[str]) -> None:
        #add a host to the known server list
        self.server_list_lock.acquire()
        self.server_list[_server_uid] = (sev_status, time.time())
        for host_type in sev_status:
            self.servernums.setdefault(host_type, 0)
            self.servernums[host_type] += 1
        self.server_list_lock.release()
        #update the known server list
        tmpdict: Dict = {
            "task_type": "control",
            "trans_c": "inavail",
            "time_stamp": time.time()
        }
        t = threading.Thread(target=self.send_tasks,
                             args=(tmpdict, _server_uid))
        t.start()
        #inform availability to the server

    def remove_host(self, server_uid: str) -> None:
        #remove the server from known server list
        self.server_list_lock.acquire()
        if server_uid in self.server_list:
            sev_status = self.server_list.pop(server_uid)[0]
            for host_type in sev_status:
                self.servernums[host_type] -= 1
                if self.servernums[host_type] == 0:
                    self.servernums.pop(host_type)
            if server_uid in self.available_server_list:
                self.available_server_list.pop(server_uid)
        self.server_list_lock.release()

        #put the tasks submitted to the server back to the queue(legacy, not sure whether removal should be done)

    def get_self_uid(self, server_uid: str) -> None:
        #online initialization
        self.remove_host(self.self_server_uid)
        self.self_server_uid = server_uid
        self.add_host(server_uid, self.self_sev_status)


class MultiMachineScheduler(RemoteTaskScheduler):

    def __init__(self,
                 addr: str = "127.0.0.1",
                 port: int = 8047,
                 sev_status: List[str] = ["default"],
                 self_server_ip="localhost",
                 self_server_port=0) -> None:
        super().__init__(sev_status)
        rpctool = core.RPCTool(scheduler=self,
                               host=addr,
                               port=port,
                               sev_status=sev_status,
                               self_server_ip=self_server_ip,
                               self_server_port=self_server_port)
        self.bind_rpctool(rpctool)
        self.get_self_uid(self.rpctool.self_host_uid)
        self.init_lock = True
