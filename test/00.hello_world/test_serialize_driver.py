import freetensor as ft
import freetensor_ffi as ffi
import numpy as np
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


def test_device_cpu():
    device = ft.Device(ft.TargetType.CPU, 0)
    target = device.target()
    txt = ft.dump_device(device)
    print(txt)
    device2 = ft.load_device(txt)
    assert device == device2


def test_device_gpu():
    device = ft.Device(ft.TargetType.GPU, 0)
    target = device.target()
    txt = ft.dump_device(device)
    print(txt)
    device2 = ft.load_device(txt)
    assert device == device2


def test_array_no_cpu_sizeof4():

    arr_np = np.array([[[1.7, 2.8, 3.9], [4.23, 5.5, 6.0]]], dtype="float32")
    arr = ft.Array(arr_np)

    head, data2 = ft.dump_array(arr)
    print(head)
    print(data2)
    arr2 = ft.load_array((head, data2))

    assert arr == arr2


def test_array_with_cpu_sizeof8():

    arr_np = np.array([[[17, 28, 7**20]], [[40, 5**24, 67]]], dtype="int64")
    arr = ft.Array(arr_np)

    head, data2 = ft.dump_array(arr)
    print(head)
    print(data2)
    arr2 = ft.load_array((head, data2))

    assert arr == arr2
