import ir
import os
import re
import random
import numpy as np
from random import randint

'''
    Profile schedule on underlying device

    Parameters
    ----------
    schedule : 
        The schedule to profile
    params : {"param_name" : param}
        The input and output parameters
    target : 
        The target device
    rounds : int
        Profile rounds, default to 2
    warmups : int
        Warmup rounds, default to 1

    Raises
    ------

    Returns
    -------
    float
        mean time cost of one round in ms
'''

def prof(schedule, params, target, rounds = 1, warmups = 0):
    ast = ir.lower(schedule.ast(), target)
    code, ir_params = ir.codegen(ast, target)
    driver = ir.Driver(code, ir_params, ir.Device(target))
    driver.set_params(params)
    tim = driver.time(rounds, warmups)
    return tim


'''
    Generate random loop spliting and reordering schedule. 

    Parameters
    ----------
    loops : [("loop_name", loop_len)]
        The loops to be scheduled. `loop_len` should be a power of 2
    ast : 
        The input ast
    target : 
        The target device


    Raises
    ------

    Returns
    -------
    schedule
        the generated schedule
'''
def gen_naive_schedule(loops, ast, target):
    s = ir.Schedule(ast)
    while True:
        target = randint(0, len(loops) - 1)
        name, size = loops[target]
        if size <= 2:
            continue
        
        facts = [2,]
        while facts[-1] * 2 < size:
            facts.append(facts[-1] * 2)
        factor = facts[randint(0, len(facts) - 1)]

        outer, inner = s.split(name, factor)
        loops[target] = (outer, size / factor)
        loops.insert(target, (inner, factor))

        if randint(0, 1) == 0:
            break
    
    random.shuffle(loops)
    s.reorder([a[0] for a in loops])
    #print(loops)
    #print(s.ast())
    return s


'''
    A naive searching strategy for autotuning. 
    Random generate loop spliting and reordering schedule and choose the best among them.

    Parameters
    ----------
    loops : [("loop_name", loop_len)]
        The loops to be scheduled. `loop_len` should be a power of 2
    ast : 
        The input ast
    params : {"param_name" : param}
        The input and output parameters
    target : 
        The target device
    iters : int
        The number of searching iterations, default to 10

    Raises
    ------

    Returns
    -------
    ast
        ast of the best schedule found
'''
def naive_search(loops, ast, params, target, iters = 10):
    best_schedule = ir.Schedule(ast)
    best_tim = prof(best_schedule, params, target)
    print(f"origin schedule time : {best_tim}")
    
    for iter in range(iters):
        curr_schedule = gen_naive_schedule(loops.copy(), ast, target)
        curr_tim = prof(curr_schedule, params, target)
        if curr_tim < best_tim:
            best_tim = curr_tim
            best_schedule = curr_schedule
        print(f"iteration : {iter + 1}/{iters}, curr time : {curr_tim}, best time : {best_tim}")

    print("best schedule : ")
    print(best_schedule.ast())
    print(f"best time {best_tim}")
    return best_schedule.ast()
