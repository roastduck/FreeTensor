from __future__ import annotations
import freetensor as ft
import threading
import copy


def measure_submit(rts: ft.RemoteTaskScheduler, tmplist):
    tmptuple = rts.remote_measure_submit(100, 10, 0, copy.copy(tmplist))
    trange = len(tmptuple[0])
    for t in range(trange):
        #assert tmptuple[0][t] == 1
        #assert tmptuple[1][t] == 0.1
        t += 1


def test_full_function():
    ft.REMOTE_TASK_SCHEDULER_GLOBAL_TEST = True
    rts = ft.RemoteTaskScheduler()

    tmplist = []
    rts.add_host("hello_test1", 3)
    rts.add_host("hello_test2", 3)
    for i in range(64):
        tmplist.append(0)

    for i in range(10):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()
