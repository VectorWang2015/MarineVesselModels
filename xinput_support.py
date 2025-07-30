"""
# encoding: utf-8
author: vectorwang@hotmail.com
last_update: 2022-04-08

reference:
https://zhuanlan.zhihu.com/p/444431769
https://docs.microsoft.com/en-us/windows/win32/xinput/getting-started-with-xinput

this module provides supports for xbox controller
"""

import ctypes
import time
from ctypes import wintypes


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', wintypes.WORD),
        ('bLeftTrigger', ctypes.c_ubyte),  # wintypes.BYTE is signed
        ('bRightTrigger', ctypes.c_ubyte),  # wintypes.BYTE is signed
        ('sThumbLX', wintypes.SHORT),
        ('sThumbLY', wintypes.SHORT),
        ('sThumbRX', wintypes.SHORT),
        ('sThumbRY', wintypes.SHORT)
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ('dwPacketNumber', wintypes.DWORD),
        ('Gamepad', XINPUT_GAMEPAD)
    ]


def _get_state(user_index):
    xinput = ctypes.windll.XInput1_4
    c_state = XINPUT_STATE()
    ret = xinput.XInputGetState(user_index, ctypes.byref(c_state))
    return ret, c_state


def _get_pedal(c_state):
    r_trigger = c_state.Gamepad.bRightTrigger # 0-255
    l_trigger = c_state.Gamepad.bLeftTrigger # 0-255
    out_data = r_trigger - l_trigger

    return int(out_data) # output [-255, 255]


def _get_l_rudder(c_state, reverse=False):
    # [-32768 , 32767]
    # aka [-0x8000 , 0x7FFF]
    raw_data = c_state.Gamepad.sThumbLX
    out_data = float(raw_data) / 32768.0 * 255.0 # [-255, 255]
    if reverse:
        out_data = -out_data
    return int(out_data)


def _get_l_thrust(c_state, reverse=False):
    # [-32768 , 32767]
    # aka [-0x8000 , 0x7FFF]
    raw_data = c_state.Gamepad.sThumbLY
    out_data = float(raw_data) / 32768.0 * 255.0 # [-255, 255]
    if reverse:
        out_data = -out_data
    return int(out_data)


def _get_r_thrust(c_state, reverse=False):
    # [-32768 , 32767]
    # aka [-0x8000 , 0x7FFF]
    raw_data = c_state.Gamepad.sThumbRY
    out_data = float(raw_data) / 32768.0 * 255.0 # [-255, 255]
    if reverse:
        out_data = -out_data
    return int(out_data)


def get_control(user_index=0, reverse=True):
    ret, state = _get_state(user_index)
    if ret != 0:
        return None
    return _get_l_thrust(state, reverse), _get_r_thrust(state, reverse)
