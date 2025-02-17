#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:17:37 2025

@author: velni
"""

import numpy as np
import sympy as sym

### r,Q sampling

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')
Q_vals = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')

### u,v,w values

uvw = np.load('/home/velni/phd/w/tfm/py/sample/uvw.npy')
uvw_str = np.load('/home/velni/phd/w/tfm/py/sample/uvw_str.npy')

uvw_full = np.load('/home/velni/phd/w/tfm/py/sample/uvw_full.npy')
uvw_str_full = np.load('/home/velni/phd/w/tfm/py/sample/uvw_str_full.npy')

### variables

u = sym.Symbol('u')
v = sym.Symbol('v')
w = sym.Symbol('w')

### functions (sympy)

def p1_pl(u,v,w):
    return 1./sym.sqrt(1**2 + u**2 + v**2 + w**2)

def m1_pl(u,v,w):
    return -1./sym.sqrt(1**2 + u**2 + v**2 + w**2)

def u_pl(u,v,w):
    return u/sym.sqrt(1**2 + u**2 + v**2 + w**2) 

def v_pl(u,v,w):
    return v/sym.sqrt(1**2 + u**2 + v**2 + w**2)

def w_pl(u,v,w):
    return w/sym.sqrt(1**2 + u**2 + v**2 + w**2)

### functions

# def ul(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     if sgn == 0:
#         y = -u/(ell**3)
#     if sgn == 1:
#         y = u/(ell**3)
#     return y

# def vl(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     if sgn == 0:
#         y = -v/(ell**3)
#     if sgn == 1:
#         y = v/(ell**3)
#     return y

# def wl(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     if sgn == 0:
#         y = -w/(ell**3)
#     if sgn == 1:
#         y = w/(ell**3)
#     return y

# #

# def uv2(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     y = (1 + u**2 + v**2)/(ell**3)
#     return y

# def uw2(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     y = (1 + u**2 + w**2)/(ell**3)
#     return y

# def vw2(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     y = (1 + v**2 + w**2)/(ell**3)
#     return y

# # 

# def uv(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     y = (-u*v)/(ell**3)
#     return y

# def uw(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     y = (-u*w)/(ell**3)
#     return y

# def vw(u,v,w,sgn):
#     ell = sym.sqrt(1 + u**2 + v**2 + w**2)
#     y = (-v*w)/(ell**3)
#     return y

### derivatives 

# def part_comb1(u,v,w,sign):
#     """ computes partial derivatives for (pm1,u,v,w)
#     row = component, column = variable
#     """
#     A = np.array([
#         [ul(u,v,w,sign),vl(u,v,w,sign),wl(u,v,w,sign)],
#         [vw2(u,v,w,sign),uv(u,v,w,sign),uw(u,v,w,sign)],
#         [uv(u,v,w,sign),uw2(u,v,w,sign),vw(u,v,w,sign)],
#         [uw(u,v,w,sign),vw(u,v,w,sign),uv2(u,v,w,sign)]
#         ])
#     return A

# def part_comb2(u,v,w,sign):
#     """ computes partial derivatives for (w,pm1,u,v)
#     row = component, column = variable
#     """
#     A = np.array([
#         [uw(u,v,w,sign),vw(u,v,w,sign),uv2(u,v,w,sign)],
#         [ul(u,v,w,sign),vl(u,v,w,sign),wl(u,v,w,sign)],
#         [vw2(u,v,w,sign),uv(u,v,w,sign),uw(u,v,w,sign)],
#         [uv(u,v,w,sign),uw2(u,v,w,sign),vw(u,v,w,sign)]
#         ])
#     return A

# def part_comb3(u,v,w,sign):
#     """ computes partial derivatives for (v,w,pm1,u)
#     row = component, column = variable
#     """
#     A = np.array([
#         [uv(u,v,w,sign),uw2(u,v,w,sign),vw(u,v,w,sign)],
#         [uw(u,v,w,sign),vw(u,v,w,sign),uv2(u,v,w,sign)],
#         [ul(u,v,w,sign),vl(u,v,w,sign),wl(u,v,w,sign)],
#         [vw2(u,v,w,sign),uv(u,v,w,sign),uw(u,v,w,sign)]
#         ])
#     return A

# def part_comb4(u,v,w,sign):
#     """ computes partial derivatives for (u,v,w,pm1)
#     row = component, column = variable
#     """
#     A = np.array([
#         [vw2(u,v,w,sign),uv(u,v,w,sign),uw(u,v,w,sign)],
#         [uv(u,v,w,sign),uw2(u,v,w,sign),vw(u,v,w,sign)],
#         [uw(u,v,w,sign),vw(u,v,w,sign),uv2(u,v,w,sign)],
#         [ul(u,v,w,sign),vl(u,v,w,sign),wl(u,v,w,sign)]
#         ])
#     return A

def part_comb1(u,v,w,sgn):
    """ computes partial derivatives for (pm1,u,v,w)
    row = component, column = variable
    """
    if sgn==0:
        A = np.array([
            [sym.diff(p1_pl(u,v,w),u),sym.diff(p1_pl(u,v,w),v),sym.diff(p1_pl(u,v,w),w)],
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)],
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)],
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)]
            ])
    if sgn==1:
        A = np.array([
            [sym.diff(m1_pl(u,v,w),u),sym.diff(m1_pl(u,v,w),v),sym.diff(m1_pl(u,v,w),w)],
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)],
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)],
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)]
            ])
    return A

def part_comb2(u,v,w,sgn):
    """ computes partial derivatives for (w,pm1,u,v)
    row = component, column = variable
    """
    if sgn==0:
        A = np.array([
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)],
            [sym.diff(p1_pl(u,v,w),u),sym.diff(p1_pl(u,v,w),v),sym.diff(p1_pl(u,v,w),w)],
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)],
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)]
            ])
    if sgn==1:
        A = np.array([
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)],
            [sym.diff(m1_pl(u,v,w),u),sym.diff(m1_pl(u,v,w),v),sym.diff(m1_pl(u,v,w),w)],
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)],
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)]
            ])
    return A

def part_comb3(u,v,w,sgn):
    """ computes partial derivatives for (v,w,pm1,u)
    row = component, column = variable
    """
    if sgn==0:
        A = np.array([
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)],
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)],
            [sym.diff(p1_pl(u,v,w),u),sym.diff(p1_pl(u,v,w),v),sym.diff(p1_pl(u,v,w),w)],
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)]
            ])
    if sgn==1:
        A = np.array([
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)],
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)],
            [sym.diff(m1_pl(u,v,w),u),sym.diff(m1_pl(u,v,w),v),sym.diff(m1_pl(u,v,w),w)],
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)]
            ])
    return A

def part_comb4(u,v,w,sgn):
    """ computes partial derivatives for (u,v,w,pm1)
    row = component, column = variable
    """
    if sgn==0:
        A = np.array([
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)],
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)],
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)],
            [sym.diff(p1_pl(u,v,w),u),sym.diff(p1_pl(u,v,w),v),sym.diff(p1_pl(u,v,w),w)]
            ])
    if sgn==1:
        A = np.array([
            [sym.diff(u_pl(u,v,w),u),sym.diff(u_pl(u,v,w),v),sym.diff(u_pl(u,v,w),w)],
            [sym.diff(v_pl(u,v,w),u),sym.diff(v_pl(u,v,w),v),sym.diff(v_pl(u,v,w),w)],
            [sym.diff(w_pl(u,v,w),u),sym.diff(w_pl(u,v,w),v),sym.diff(w_pl(u,v,w),w)],
            [sym.diff(m1_pl(u,v,w),u),sym.diff(m1_pl(u,v,w),v),sym.diff(m1_pl(u,v,w),w)]
            ])
    return A

### diferentials 

def dA_dB_dC(u,v,w,A,B,C,comb,sgn):
    """ computes 3-form for a given combination
    """
    
    # choose comb+sgn
    if comb==1:
        mtx = part_comb1(u,v,w,sgn)
    if comb==2:
        mtx = part_comb2(u,v,w,sgn)
    if comb==3:
        mtx = part_comb3(u,v,w,sgn)
    if comb==4:
        mtx = part_comb4(u,v,w,sgn)
    
    # derivatives
    pA = mtx[A,:]
    pB = mtx[B,:]
    pC = mtx[C,:]
    
    # 3-form
    d = pA[0]*pB[1]*pC[2] - pA[0]*pB[2]*pC[1] - pA[1]*pB[0]*pC[2] + pA[1]*pB[2]*pC[0] + pA[2]*pB[0]*pC[1] - pA[2]*pB[1]*pC[0]
        
    du = 1.
    dv = 1.
    dw = 1.
    
    return d*du*dv*dw

### dQ

def dQ(u,v,w,comb,sgn):
    """ compute dQ for given (u,v,w) and comb+sign
    """
    
    ell_inv = 1/sym.sqrt(1 + u**2 + v**2 + w**2)
    
    d0 = dA_dB_dC(u,v,w,1,2,3,comb,sgn)
    d1 = dA_dB_dC(u,v,w,0,2,3,comb,sgn)
    d2 = dA_dB_dC(u,v,w,0,1,3,comb,sgn)
    d3 = dA_dB_dC(u,v,w,0,1,2,comb,sgn)
    
    if comb==1: # (pm1,u,v,w)
        dQ = ell_inv*(((-1)**sgn)*d0 - u*d1 + v*d2 - w*d3)
    
    if comb==2: # (w,pm1,u,v)
        dQ = ell_inv*(w*d0 - ((-1)**sgn)*d1 + u*d2 - v*d3)
        
    if comb==3: # (v,w,pm1,u)
        dQ = ell_inv*(v*d0 - w*d1 + ((-1)**sgn)*d2 - u*d3)
        
    if comb==4: # (u,v,w,pm1)
        dQ = ell_inv*(u*d0 - v*d1 + w*d2 - ((-1)**sgn)*d3)
    
    return dQ

#%% analytical results

dQ_10 = dQ(u,v,w,1,0)
dQ_11 = dQ(u,v,w,1,1)
   
dQ_20 = dQ(u,v,w,2,0)
dQ_21 = dQ(u,v,w,2,1)

dQ_30 = dQ(u,v,w,3,0)
dQ_31 = dQ(u,v,w,3,1)

dQ_40 = dQ(u,v,w,4,0)
dQ_41 = dQ(u,v,w,4,1)

dQ_an = [sym.simplify(dQ_10), sym.simplify(dQ_11), sym.simplify(dQ_20), sym.simplify(dQ_21), sym.simplify(dQ_30), sym.simplify(dQ_31), sym.simplify(dQ_40), sym.simplify(dQ_41)]

#%% numerical results

dQ_vals = [[] for i in range(8)]

for c_idx,c in enumerate(uvw_str_full):
    for i in range(4):
        if c[i] == str('1'):
            comb = i+1
            sgn = 0
        if c[i] == str('-1'):
            comb = i+1
            sgn = 1
        else:
            pass

    if comb==1 and sgn==0:
        k = 0
    if comb==1 and sgn==1:
        k = 1

    if comb==2 and sgn==0:
        k = 2
    if comb==2 and sgn==1:
        k = 3
        
    if comb==3 and sgn==0:
        k = 4
    if comb==3 and sgn==1:
        k = 5
        
    if comb==4 and sgn==0:
        k = 6
    if comb==4 and sgn==1:
        k = 7
    
    xu = (1 + comb-1) % 4
    xv = (2 + comb-1) % 4
    xw = (3 + comb-1) % 4
    
    dQ_vals[k].append( dQ_an[k].evalf(subs={u:uvw_full[c_idx][xu],v:uvw_full[c_idx][xv],w:uvw_full[c_idx][xw]}) )

#%% check

# dQ
vol1 = sum(dQ_vals[0])

# Euler angles
x,t,f = sym.symbols("x t f")
arg = (sym.sin(x)**2)*sym.sin(t)
vol2 = sym.integrate(arg, (x, 0, sym.pi), (t, 0, sym.pi), (f, 0, 2*sym.pi))
