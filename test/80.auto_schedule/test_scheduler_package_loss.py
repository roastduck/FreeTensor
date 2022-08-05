import freetensor as ft
import threading
import copy


def measure_submit(rts: ft.RemoteTaskScheduler, tmplist):
    tmptuple = rts.remote_measure_submit(100, 10, 0, copy.copy(tmplist))
    trange = len(tmptuple[0])
    for t in range(trange):
        assert tmptuple[0][t] == 1
        assert tmptuple[1][t] == 0.1


def test_full_function_with_package_loss():
    rts = ft.RemoteTaskScheduler()
    rts.change_into_test_mode()
    rts.config_package_loss_rate(0.03)

    tmplist = []

    for i in range(64):
        tmplist.append(0)

    for i in range(1000):
        thread_test = threading.Thread(target=measure_submit,
                                       args=(rts, tmplist))
        thread_test.start()

    print(rts.recalls)
