import freetensor as ft
import pytest
from random import randint

def test_basic():
    l, r = -2147483648, 2147483647
    for i in range(10000):
        # CPU
        target = ft.CPU(randint(0, 1))
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2

        device = ft.Device(target, randint(0, r))
        txt = ft.dump_device(device)
        print(txt)
        device2 = ft.load_device(txt)
        assert device == device2

        # GPU
        target = ft.GPU(randint(0, 1))
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2

        device = ft.Device(target, randint(0, r))
        txt = ft.dump_device(device)
        print(txt)
        device2 = ft.load_device(txt)
        assert device == device2

        # GPU with compute_capability
        target = ft.GPU(randint(0, 1))
        target.set_compute_capability(randint(l, r), randint(l, r))
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2

        device = ft.Device(target, randint(0, r))
        txt = ft.dump_device(device)
        print(txt)
        device2 = ft.load_device(txt)
        assert device == device2




