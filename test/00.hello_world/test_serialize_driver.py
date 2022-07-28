import freetensor as ft
import pytest
from random import randint


def test_target_cpu():
    target = ft.CPU(0)
    txt = ft.dump_target(target)
    print(txt)
    target2 = ft.load_target(txt)
    assert target == target2


def test_target_gpu():
    target = ft.GPU(0)
    txt = ft.dump_target(target)
    print(txt)
    target2 = ft.load_target(txt)
    assert target == target2


def test_target_gpu_with_compute_capability():
    target = ft.GPU(0)
    target.set_compute_capability(7, 0)
    txt = ft.dump_target(target)
    print(txt)
    target2 = ft.load_target(txt)
    assert target == target2

    target = ft.GPU(1)
    target.set_compute_capability(2147483647, -2147483648)
    txt = ft.dump_target(target)
    print(txt)
    target2 = ft.load_target(txt)
    assert target == target2


def test_device_cpu():
    target = ft.CPU(1)
    device = ft.Device(target, 2)
    txt = ft.dump_device(device)
    print(txt)
    device2 = ft.load_device(txt)
    assert device == device2


def test_device_gpu():
    target = ft.GPU(1)
    device = ft.Device(target, 1)
    txt = ft.dump_device(device)
    print(txt)
    device2 = ft.load_device(txt)
    assert device == device2


def test_device_gpu_with_compute_capability():
    target = ft.GPU(1)
    target.set_compute_capability(7, 0)
    device = ft.Device(target, 0)
    txt = ft.dump_device(device)
    print(txt)
    device2 = ft.load_device(txt)
    assert device == device2
