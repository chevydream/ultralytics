#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Project -> File: main.py -> global_var.py
@author: ***
@file: global_var.py
@time: 2023/12/5 13:55
@desc: 单独拿一个py文件来单独存放全局变量（https://zhuanlan.zhihu.com/p/349108535）
        其他文件需要用到的，则import global_var.py。
        在主文件初始化一下，global_var._init()
        写入值：global_var.set_value('stop_flag',stop_flag)
        读取值：input_step = global_var.get_value('input_step')
"""


def _init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    # 定义一个全局变量
    _global_dict[key] = value


def get_value(key, defValue=None):
    # 获得一个全局变量，不存在则提示读取对应变量失败，也可以返回默认值
    try:
        return _global_dict[key]
    except:
        # print('读取'+key+'失败\r\n')  # 提示读取对应变量失败
        return defValue  # 返回默认值