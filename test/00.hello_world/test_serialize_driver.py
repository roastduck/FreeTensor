import freetensor as ft
import pytest
from random import randint


def test_target_cpu():
    for use_native_arch in range(2):
        target = ft.CPU(use_native_arch)
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2


def test_target_gpu():
    for use_native_arch in range(2):
        target = ft.GPU(use_native_arch)
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2


def test_target_gpu_with_compute_capability():
    for use_native_arch in range(2):
        target = ft.GPU(use_native_arch)
        target.set_compute_capability(7, 0)
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2

        target = ft.GPU(use_native_arch)
        target.set_compute_capability(2147483647, -2147483648)
        txt = ft.dump_target(target)
        print(txt)
        target2 = ft.load_target(txt)
        assert target == target2


def test_device_cpu():
    for device_num in range(4):
        for use_native_arch in range(2):
            target = ft.CPU(use_native_arch)
            device = ft.Device(target, device_num)
            txt = ft.dump_device(device)
            print(txt)
            device2 = ft.load_device(txt)
            assert device == device2


def test_device_gpu():
    for device_num in range(4):
        for use_native_arch in range(2):
            target = ft.GPU(use_native_arch)
            device = ft.Device(target, device_num)
            txt = ft.dump_device(device)
            print(txt)
            device2 = ft.load_device(txt)
            assert device == device2


def test_device_gpu_with_compute_capability():
    for device_num in range(4):
        for use_native_arch in range(2):
            target = ft.GPU(use_native_arch)
            target.set_compute_capability(7, 0)
            device = ft.Device(target, device_num)
            txt = ft.dump_device(device)
            print(txt)
            device2 = ft.load_device(txt)
            assert device == device2
