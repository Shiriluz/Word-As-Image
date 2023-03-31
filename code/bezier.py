import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from numpy.linalg import norm

def num_bezier(n_ctrl, degree=3):
    if type(n_ctrl) == np.ndarray:
        n_ctrl = len(n_ctrl)
    return int((n_ctrl - 1) / degree)

def bernstein(n, i):
    bi = binom(n, i)
    return lambda t, bi=bi, n=n, i=i: bi * t**i * (1 - t)**(n - i)

def bezier(P, t, d=0):
    '''Bezier curve of degree len(P)-1. d is the derivative order (0 gives positions)'''
    n = P.shape[0] - 1
    if d > 0:
        Q = np.diff(P, axis=0)*n
        return bezier(Q, t, d-1)
    B = np.vstack([bernstein(n, i)(t) for i, p in enumerate(P)])
    return (P.T @ B).T

def cubic_bezier(P, t):
    return (1.0-t)**3*P[0] + 3*(1.0-t)**2*t*P[1] + 3*(1.0-t)*t**2*P[2] + t**3*P[3]

def bezier_piecewise(Cp, subd=100, degree=3, d=0):
    ''' sample a piecewise Bezier curve given a sequence of control points'''
    num = num_bezier(Cp.shape[0], degree)
    X = []
    for i in range(num):
        P = Cp[i*degree:i*degree+degree+1, :]
        t = np.linspace(0, 1., subd)[:-1]
        Y = bezier(P, t, d)
        X += [Y]
    X.append(Cp[-1])
    X = np.vstack(X)
    return X

def compute_beziers(beziers, subd=100, degree=3):
    chain = beziers_to_chain(beziers)
    return bezier_piecewise(chain, subd, degree)

def plot_control_polygon(Cp, degree=3, lw=0.5, linecolor=np.ones(3)*0.1):
    n_bezier = num_bezier(len(Cp), degree)
    for i in range(n_bezier):
        cp = Cp[i*degree:i*degree+degree+1, :]
        if degree==3:
            plt.plot(cp[0:2,0], cp[0:2, 1], ':', color=linecolor, linewidth=lw)
            plt.plot(cp[2:,0], cp[2:,1], ':', color=linecolor, linewidth=lw)
            plt.plot(cp[:,0], cp[:,1], 'o', color=[0, 0.5, 1.], markersize=4)
        else:
            plt.plot(cp[:,0], cp[:,1], ':', color=linecolor, linewidth=lw)
            plt.plot(cp[:,0], cp[:,1], 'o', color=[0, 0.5, 1.])


def chain_to_beziers(chain, degree=3):
    ''' Convert Bezier chain to list of curve segments (4 control points each)'''
    num = num_bezier(chain.shape[0], degree)
    beziers = []
    for i in range(num):
        beziers.append(chain[i*degree:i*degree+degree+1,:])
    return beziers


def beziers_to_chain(beziers):
    ''' Convert list of Bezier curve segments to a piecewise bezier chain (shares vertices)'''
    n = len(beziers)
    chain = []
    for i in range(n):
        chain.append(list(beziers[i][:-1]))
    chain.append([beziers[-1][-1]])
    return np.array(sum(chain, []))


def split_cubic(bez, t):
    p1, p2, p3, p4 = bez

    p12 = (p2 - p1) * t + p1
    p23 = (p3 - p2) * t + p2
    p34 = (p4 - p3) * t + p3

    p123 = (p23 - p12) * t + p12
    p234 = (p34 - p23) * t + p23

    p1234 = (p234 - p123) * t + p123

    return np.array([p1, p12, p123, p1234]), np.array([p1234, p234, p34, p4])


def approx_arc_length(bez):
    c0, c1, c2, c3 = bez
    v0 = norm(c1-c0)*0.15
    v1 = norm(-0.558983582205757*c0 + 0.325650248872424*c1 + 0.208983582205757*c2 + 0.024349751127576*c3)
    v2 = norm(c3-c0+c2-c1)*0.26666666666666666
    v3 = norm(-0.024349751127576*c0 - 0.208983582205757*c1 - 0.325650248872424*c2 + 0.558983582205757*c3)
    v4 = norm(c3-c2)*.15
    return v0 + v1 + v2 + v3 + v4


def subdivide_bezier(bez, thresh):
    stack = [bez]
    res = []
    while stack:
        bez = stack.pop()
        l = approx_arc_length(bez)
        if l < thresh:
            res.append(bez)
        else:
            b1, b2 = split_cubic(bez, 0.5)
            stack += [b2, b1]
    return res

def subdivide_bezier_chain(C, thresh):
    beziers = chain_to_beziers(C)
    res = []
    for bez in beziers:
        res += subdivide_bezier(bez, thresh)
    return beziers_to_chain(res)



