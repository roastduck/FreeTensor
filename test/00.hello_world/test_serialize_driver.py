import freetensor as ft
import freetensor_ffi as ffi
from freetensor import CPU, GPU, Device
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


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_array_no_cpu_sizeof4():
    data = '#$^!@*ABCDEF114514ghijkl'
    assert len(data) == 24

    target = GPU(1)
    target.set_compute_capability(7, 0)
    devs = [Device(GPU(1), 1), Device(GPU(0), 2), Device(target, 3)]

    arr = ffi.new_test_array([1, 2, 3], "float32", devs, data)
    head, data2 = ft.dump_array(arr)
    print(head)
    print(data2)
    print(type(data2))
    arr2 = ft.load_array(head, data2)

    assert arr == arr2


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_array_with_cpu_sizeof8():
    data = '#$^!@*ABCDEF114514ghijkl(&%,.?MNOPQR236789stuvwx'
    assert len(data) == 48

    target = GPU(0)
    target.set_compute_capability(4, 3)
    devs = [Device(CPU(1), 1), Device(CPU(0), 2), Device(target, 3)]

    arr = ffi.new_test_array([2, 1, 3], "int64", devs, data)
    head, data2 = ft.dump_array(arr)
    print(head)
    print(data2)
    print(type(data2))
    arr2 = ft.load_array(head, data2)

    assert arr == arr2
