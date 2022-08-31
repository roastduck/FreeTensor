import freetensor as ft
import threading
import copy
import pytest
from typing import List, Dict, Any
import time
import random


class MeasureTaskTest(ft.MeasureTask):

    def measure(self, rounds: int, warmups: int, attached_params: tuple,
                sketches: List) -> tuple[List[float], List[float]]:
        tmplist1 = []
        tmplist2 = []
        #time.sleep(1)  #reserved for realistic tests
        for k in range(len(sketches)):
            tmplist1.append(1.0)
            tmplist2.append(0.1)
        t = (tmplist1, tmplist2)
        return t

    @classmethod
    def dict2self(cls, inputdict: Dict) -> Any:
        tmptask = super().dict2self_base(
            MeasureTaskTest(0, 0, inputdict["warmup_rounds"],
                            inputdict["attached_params"], []), inputdict)
        return tmptask


class RemoteTaskSchedulerTest(ft.RemoteTaskScheduler):
    task_internet_delay: float = 0.0
    result_internet_delay: float = 0.0
    package_loss_rate: float = 0.0

    def __init__(self,
                 _task_internet_delay=0.0,
                 _result_internet_delay=0.0,
                 _package_loss_rate=0.0) -> None:
        super().__init__()
        self.task_internet_delay = _task_internet_delay
        self.result_internet_delay = _result_internet_delay
        self.package_loss_rate = _package_loss_rate

    def remote_measure_submit(
            self, rounds: int, warmups: int, attached_params: int,
            Sketches: List[str]) -> tuple[List[float], List[float]]:
        #the method used to submit measure task by cpp-python-bridge
        tmpuid = self.task_uid_assign()
        tmptask = MeasureTaskTest(tmpuid, rounds, warmups, attached_params,
                                  Sketches)
        tmptask.host_type = "default"
        tmplock = threading.Event()
        self.task_register(tmptask, tmplock)
        tmplock.wait()
        tmpresult = self.task_result_container.pop(tmpuid).fetch()
        return tmpresult

    def send_tasks(self, _task: Dict, server_uid: str) -> int:
        _task.setdefault("time_stamp", time.time())
        time.sleep(self.task_internet_delay)
        t = random.random()
        if t < self.package_loss_rate:
            return -1
        return self.remote_task_receive(self.self_server_uid, _task)
        #this part will use the method in RPCTools

    def send_results(self, _taskresult: Dict, server_uid: str) -> None:
        time.sleep(self.result_internet_delay)
        t = random.random()
        if t < self.package_loss_rate:
            return
        if self.verbose > 0:
            print("sending results to" + server_uid)
        if server_uid == "localhost":
            self.remote_result_receive(self.self_server_uid, _taskresult)
            return
        else:
            self.rpctool.remote_result_submit(server_uid, _taskresult)

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
                tmptask = MeasureTaskTest.dict2self(task)

            self.task_execution(tmptask)
        return 0


def measure_submit(rts: ft.RemoteTaskScheduler, tmplist):
    tmptuple = rts.remote_measure_submit(100, 10, 0, copy.copy(tmplist))
    trange = len(tmptuple[0])
    for t in range(trange):
        assert tmptuple[0][t] == 1
        assert tmptuple[1][t] == 0.1


def test_full_function():
    rts = RemoteTaskSchedulerTest()
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(64):
        tmplist.append(0)

    for i in range(1000):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()
        tmpthreadlist.append(thread_test)

    for t in tmpthreadlist:
        t.join()
    print(rts.recalls)


def test_full_function_with_delay():
    rts = RemoteTaskSchedulerTest(
        0.4,
        0.4,
    )
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(64):
        tmplist.append(0)

    for i in range(400):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()
        tmpthreadlist.append(thread_test)

    for t in tmpthreadlist:
        t.join()
    print(rts.recalls)


@pytest.mark.skip()
def test_full_function_stress_test():
    rts = RemoteTaskSchedulerTest(0.04, 0.04, 0)
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(64):
        tmplist.append(0)

    for i in range(1):
        for j in range(100):
            thread_test = threading.Thread(target=measure_submit,
                                           args=(rts, tmplist))
            thread_test.start()
            tmpthreadlist.append(thread_test)

            for t in tmpthreadlist:
                t.join()

            tmpthreadlist = []
        print((rts.recalls, rts.inavailability_counter))


@pytest.mark.skip()
def test_full_function_with_package_loss():
    rts = RemoteTaskSchedulerTest(0, 0, 0.01)
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(64):
        tmplist.append(0)

    for i in range(1000):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()
        tmpthreadlist.append(thread_test)

    for t in tmpthreadlist:
        t.join()
    print(rts.recalls)


@pytest.mark.skip()
def test_full_function_with_delay_and_package_loss():
    rts = RemoteTaskSchedulerTest(0.04, 0.04, 0.01)
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(64):
        tmplist.append(0)

    for i in range(400):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()
        tmpthreadlist.append(thread_test)

    for t in tmpthreadlist:
        t.join()
    print(rts.recalls)
