import freetensor as ft
import threading
import copy
import pytest


def measure_submit(rts: ft.RemoteTaskScheduler, tmplist):
    tmptuple = rts.remote_measure_submit(100, 10, 0, copy.copy(tmplist))
    trange = len(tmptuple[0])
    for t in range(trange):
        assert tmptuple[0][t] == 1
        assert tmptuple[1][t] == 0.1


def test_full_function():
    ft.RemoteTaskScheduler.change_into_test_mode()
    ft.RemoteTaskScheduler.config_package_loss_rate(0.00)
    ft.RemoteTaskScheduler.config_transmittion_delay(0.00)
    rts = ft.RemoteTaskScheduler()
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
    ft.RemoteTaskScheduler.change_into_test_mode()
    ft.RemoteTaskScheduler.config_package_loss_rate(0.0)
    ft.RemoteTaskScheduler.config_transmittion_delay(0.04)
    rts = ft.RemoteTaskScheduler()
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
    ft.RemoteTaskScheduler.change_into_test_mode()
    ft.RemoteTaskScheduler.config_package_loss_rate(0.0)
    rts = ft.RemoteTaskScheduler()
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
