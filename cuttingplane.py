#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import copy
import scipy.optimize
import scipy.spatial
import time

def rej_sample(n, constraints, b_ub):
    samples = np.empty((0, dim))
    keep_idx = lambda sample: ((constraints @ sample.T).T <= b_ub).all(axis=1)
    while samples.shape[0] < n:
        candidate = np.random.random((n, dim))
        samples = np.append(samples, candidate[keep_idx(candidate), :], axis=0)

    return np.array(samples[0:n, :])


def get_feasible(constraints, b_ub):
    # Finds feasible point in the convex hull (centre point of largest sphere)
    c = np.zeros((constraints.shape[1] + 1,))

    c[-1] = -1
    norm_vector = np.linalg.norm(constraints, axis=1)[:, None]
    A = np.hstack((constraints, norm_vector))
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b_ub, bounds=(None, None), method='interior-point')
    feasible_point = np.array(res.x[:-1])
    return feasible_point


def plot_all(
    constraints,
    b_ub,
    optimum=None,
    belt=None,
    filename="cuttingplane.pdf",
    limits="simplex",
):
    dim = constraints.shape[-1]
    if dim == 1:
        return
    fig, axs = plt.subplots(dim, dim, figsize=(15, 15), sharex=True, sharey=True)

    # HalfspaceIntersection uses Ax-b<=0
    halfspaces = np.concatenate((constraints, -b_ub[:, None]), axis=1)

    feasible_point = get_feasible(constraints, b_ub)

    hs = scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point)

    for i in range(dim):
        for j in range(dim):
            if i != j:

                if optimum is not None:
                    axs[j, i].plot([optimum[i]], [optimum[j]], marker="*", color="k")
                if belt is not None:
                    axs[j, i].plot([belt[i]], [belt[j]], marker="x")
                # axs[j,i].scatter(samples[:,i], samples[:,j]
                intersect_points = np.concatenate(
                    (hs.intersections[:, i][:, None], hs.intersections[:, j][:, None]),
                    axis=1,
                )
                hull = scipy.spatial.ConvexHull(intersect_points)
                for simplex in hull.simplices:
                    axs[j, i].plot(
                        intersect_points[simplex, 0], intersect_points[simplex, 1], "k-"
                    )

                if limits == "simplex":
                    axs[j, i].set_xlim([0, 1])
                    axs[j, i].set_ylim([0, 1])

    plt.savefig(filename)
    plt.close()


def cutting_plane(dim, constraints, b_ub, epsilon, delta, tangent, steps, samples, plot=False):
    for i in range(steps):
        t0 = time.time()
        data = hr_sampling(samples, constraints, b_ub)
        beta_t = np.mean(data, axis=0)
        A = np.mean(data[:, :, None] @ data[:, None, :], axis=0)
        B = scipy.linalg.sqrtm(scipy.linalg.inv(A))
        C = tangent(epsilon, beta_t)
        Cnorm = scipy.linalg.norm(C)
        if epsilon == 0: #Numerically better?
            c = C
        else:
            c = C / Cnorm
        #print("Bnorm: ", scipy.linalg.norm(B))
        #print("Cnorm: ", Cnorm)
        #print("B/C norm:", scipy.linalg.norm(B@c)/Cnorm)
        if Cnorm < delta:
            return {
                "belief": beta_t,
                "term": "delta",
                "delta": delta,
                "constraints": constraints,
                "b_ub": b_ub,
            }
        b = c @ beta_t + 2 * epsilon  # Check signs here
        constraints = np.concatenate((constraints, c[None, :]), axis=0)
        b_ub = np.concatenate((b_ub, [b]))
        beta_t = np.append(beta_t,  1-np.sum(beta_t))
        print("beta:", beta_t)
        out = {"constraints": constraints, "b_ub": b_ub, "term": "iter", "belief": beta_t}
        if plot:
            plot_all(out["constraints"], out["b_ub"], limits="simplex", filename="planes/cuttingplane{}.pdf".format(i))
        print("Time", time.time()-t0)
    return out


def hr_sampling(n, constraints, b_ub, burnin=100):
    dim = constraints.shape[-1]
    x = []
    l0 = get_feasible(constraints, b_ub)
    # Not optimal burnin, but only impacts convergence speed
    random_vectors = np.random.normal(size=(n + burnin, dim))
    random_vectors = random_vectors / scipy.linalg.norm(random_vectors, axis=1)[:, None]
    for i in range(n + burnin):
        min_d = np.finfo(np.float64).max
        direction = random_vectors[i, :]
        # Notation from https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        for c, b in zip(constraints, b_ub):
            p0 = np.zeros(dim)
            idx = np.argmax(np.abs(c))
            p0[idx] = b / c[idx]
            d = (p0 - l0) @ c / (direction @ c)
            if d > 0 and d < min_d:
                min_d = d
        newpoint = l0 + np.random.random() * min_d * direction
        x.append(newpoint)
        l0 = newpoint
    return np.array(x)[-n:]


def get_initial_constraints(dim):
    initial_constraints = -np.identity(dim)
    initial_constraints = np.concatenate(
        (initial_constraints, np.ones((1, dim))), axis=0
    )
    initial_b_ub = [0] * dim
    initial_b_ub.append(1)
    initial_b_ub = np.array(initial_b_ub)
    return initial_constraints, initial_b_ub


def get_initial_constraints_beta_ab():
    dim = 2
    initial_constraints = -np.identity(dim)
    initial_constraints = np.concatenate(
        (initial_constraints, np.ones((1, dim))), axis=0
    )
    initial_b_ub = [0] * dim
    initial_b_ub.append(100)
    initial_b_ub = np.array(initial_b_ub)
    return initial_constraints, initial_b_ub


if __name__ == "__main__":
    dim = 8
    optimum = [0.1, 0.3, 0.15, 0.5, 0.05, 0.15, 0.5, 0.4]
    optimum = optimum[0:dim]
    optimum = optimum / np.sum(optimum)
    assert np.sum(optimum) < 1.0000001
    assert np.sum(optimum) > 0.9999999

    epsilon = 0.1
    delta = 0.1

    def fun(x):
        return (x - optimum) ** 2

    def sample_fun(x, epsilon):
        return fun(x) - np.random.random() * epsilon

    def tangent(epsilon, beta):
        return 2 * (beta - optimum) - np.random.random(dim) * epsilon

    constraints, b_ub = get_initial_constraints(dim)
    out = cutting_plane(dim, constraints, b_ub, epsilon, delta, tangent, 25, 10000)

    plot_all(out["constraints"], out["b_ub"], optimum, out["belief"])
