import freetensor as ft
import threading
import copy
import time


def measure_submit(rts: ft.RemoteTaskScheduler, tmplist):
    tmptuple = rts.remote_measure_submit(100, 10, 0, copy.copy(tmplist))
    trange = len(tmptuple[0])
    for t in range(trange):
        assert tmptuple[0][t] == 1
        assert tmptuple[1][t] == 0.1


def test_full_function_with_package_loss():
    ft.RemoteTaskScheduler.change_into_test_mode()
    ft.RemoteTaskScheduler.config_package_loss_rate(0.01)
    ft.RemoteTaskScheduler.config_transmittion_delay(0.0)
    rts = ft.RemoteTaskScheduler()
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(640):
        tmplist.append(0)

    for i in range(10):
        for j in range(1):
            thread_test = threading.Thread(target=measure_submit,
                                           args=(rts, tmplist))
            thread_test.start()
            tmpthreadlist.append(thread_test)

            for t in tmpthreadlist:
                t.join()

            tmpthreadlist = []
        print((rts.recalls, rts.inavailability_counter))
        time.sleep(0.1)


def test_full_function_with_delay_and_package_loss():
    ft.RemoteTaskScheduler.change_into_test_mode()
    ft.RemoteTaskScheduler.config_package_loss_rate(0.01)
    ft.RemoteTaskScheduler.config_transmittion_delay(0.04)
    rts = ft.RemoteTaskScheduler()
    rts.verbose = 0
    tmplist = []
    tmpthreadlist = []
    for i in range(64):
        tmplist.append(0)

    for i in range(10):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()
        tmpthreadlist.append(thread_test)

    for t in tmpthreadlist:
        t.join()
    print(rts.recalls)
