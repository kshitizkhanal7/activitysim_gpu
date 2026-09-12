"""Compiled, parallel CPU algorithm controls (no pandas inside timed loops).

These are controls for departure retries and nested reduction, not a claim to
replace all ActivitySim CPU computation. Fast-math is deliberately disabled.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def choose_cpu(p, base, lo, hi, z, norm):
    total = 0.
    for a in range(len(p)):
        if lo <= a+base <= hi:
            total += p[a]
    fail = (0. if total > 0 else 1.) if norm else 1.-min(1., max(0., total))
    largest, best = -1., len(p)
    for a in range(len(p)):
        v = p[a] if lo <= a+base <= hi else 0.
        if norm and total > 0:
            v /= total
        if v > largest:
            largest, best = v, a
        z -= v
        if z <= 0:
            return a+base
    if fail > largest:
        best = len(p)
    z -= fail
    a = len(p) if z <= 0 else best
    return -1 if a == len(p) else a+base


@njit(cache=True, parallel=True)
def retries_cpu(ptr, outbound, num, count, fixed, hour, lo, hi, specrow, drawrow,
                draws, probs, base, firstout, firstin, iterations):
    result = np.full(len(num), -1, np.int32)
    used = np.zeros(len(num), np.int32)
    failures = np.zeros(len(ptr)-1, np.int32)
    for g in prange(len(ptr)-1):
        begin, end = ptr[g], ptr[g+1]
        activeout, activein = True, True
        for i in range(iterations):
            if not activeout and not activein:
                break
            maxout, previous = -1, -1
            badout, badin = False, False
            if activeout:
                for r in range(begin, end):
                    if not outbound[r]:
                        continue
                    lower = previous if previous >= 0 else lo[r]
                    if fixed[r]:
                        v = hour[r]
                    else:
                        used[r] += 1
                        v = choose_cpu(probs[specrow[r]], base, lower, hi[r],
                                       draws[drawrow[r], i], num[r] == firstout)
                        if v < 0:
                            failures[g] += 1
                            badout = True
                            if i == iterations-1:
                                v = lower
                        previous = v if v >= 0 else lower
                    result[r] = v
                    maxout = max(maxout, v)
            previous = -1
            if activein:
                for r in range(end-1, begin-1, -1):
                    if outbound[r]:
                        continue
                    upper = previous if previous >= 0 else hi[r]
                    lower = maxout if maxout >= 0 else lo[r]
                    if fixed[r]:
                        v = hour[r]
                    else:
                        used[r] += 1
                        v = choose_cpu(probs[specrow[r]], base, lower, upper,
                                       draws[drawrow[r], i], count[r]-num[r] == firstin)
                        if v < 0:
                            failures[g] += 1
                            badin = True
                            if i == iterations-1:
                                v = upper
                        previous = v if v >= 0 else upper
                    result[r] = v
            activeout, activein = badout, badin
    return result, used, failures


@njit(cache=True)
def sum_cpu(u, a, b, clip=False):
    total = 0.
    for i in range(a, b):
        total += 0. if clip and u[i] <= 1e-300 else u[i]
    return total


@njit(cache=True)
def ratio_cpu(v, den):
    return min(1., (0. if v <= 1e-300 else v)/den) if den > 0 else 0.


@njit(cache=True, parallel=True, error_model="numpy")
def modes_cpu(raw, draws, c):
    rows = len(raw)
    choices = np.empty(rows, np.int32)
    logsums = np.empty(rows, np.float64)
    guards = np.zeros(rows, np.uint8)
    for row in prange(rows):
        u, p = np.zeros(31, np.float64), np.empty(21, np.float64)
        for a in range(21):
            scale = c[0]*c[1] if a < 6 else (c[2] if a < 8 else (c[3]*c[4] if a < 18 else c[5]))
            u[a] = math.exp(np.float64(raw[row, a])/scale)
        u[21] = math.exp(c[1]*np.log(sum_cpu(u, 0, 2)))
        u[22] = math.exp(c[1]*np.log(sum_cpu(u, 2, 4)))
        u[23] = math.exp(c[1]*np.log(sum_cpu(u, 4, 6)))
        u[24] = math.exp(c[0]*np.log(sum_cpu(u, 21, 24)))
        u[25] = math.exp(c[2]*np.log(sum_cpu(u, 6, 8)))
        u[26] = math.exp(c[4]*np.log(sum_cpu(u, 8, 13)))
        u[27] = math.exp(c[4]*np.log(sum_cpu(u, 13, 18)))
        u[28] = math.exp(c[3]*np.log(sum_cpu(u, 26, 28)))
        u[29] = math.exp(c[5]*np.log(sum_cpu(u, 18, 21)))
        root_raw = ((u[24]+u[25])+u[28])+u[29]
        root_den = (sum_cpu(u, 24, 26, True)+(0. if u[28] <= 1e-300 else u[28]))+(0. if u[29] <= 1e-300 else u[29])
        logsums[row] = np.log(math.exp(np.log(root_raw)))
        au, nm = ratio_cpu(u[24], root_den), ratio_cpu(u[25], root_den)
        tr, rh = ratio_cpu(u[28], root_den), ratio_cpu(u[29], root_den)
        for a in range(6):
            begin, parent = (a//2)*2, 21+a//2
            p[a] = (au*ratio_cpu(u[parent], sum_cpu(u, 21, 24, True)))*ratio_cpu(u[a], sum_cpu(u, begin, begin+2, True))
        for a in range(6, 8):
            p[a] = nm*ratio_cpu(u[a], sum_cpu(u, 6, 8, True))
        for a in range(8, 18):
            begin, parent = (8, 26) if a < 13 else (13, 27)
            p[a] = (tr*ratio_cpu(u[parent], sum_cpu(u, 26, 28, True)))*ratio_cpu(u[a], sum_cpu(u, begin, begin+5, True))
        for a in range(18, 21):
            p[a] = rh*ratio_cpu(u[a], sum_cpu(u, 18, 21, True))
        z, maximum, selected, maxpos = draws[row], -1., -1, 0
        for a in range(21):
            if p[a] > maximum:
                maximum, maxpos = p[a], a
            z -= p[a]
            if abs(z) <= 1e-9:
                guards[row] = 1
            if selected < 0 and z <= 0:
                selected = a
        choices[row] = maxpos if selected < 0 else selected
    return choices, logsums, guards
