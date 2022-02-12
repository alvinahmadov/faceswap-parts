# License (Modified BSD)
# Copyright (C) 2011, the scikit-image team All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
# Neither the name of skimage nor the names of its contributors may be used to endorse
# or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# umeyama function from scikit-image/skimage/transform/_geometric.py

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool) -> np.ndarray:
    """
    Umeyama algorithm implementation.

    Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
     Source coordinates.
    dst : (M, N) array
     Destination coordinates
    estimate_scale : bool
     Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    https://web.stanford.edu/class/cs273/refs/umeyama.pdf

    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num: int = src.shape[0]
    dim: int = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    a = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(a) < 0:
        d[dim - 1] = -1

    t = np.eye(dim + 1, dtype=np.double)

    u, s, v = np.linalg.svd(a)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        return np.nan * t
    elif rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(v) > 0:
            t[:dim, :dim] = np.dot(u, v)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            t[:dim, :dim] = np.dot(u, np.dot(np.diag(d), v))
            d[dim - 1] = s
    else:
        t[:dim, :dim] = np.dot(u, np.dot(np.diag(d), v.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(s, d)
    else:
        scale = 1.0

    t[:dim, dim] = dst_mean - scale * np.dot(t[:dim, :dim], src_mean.T)
    t[:dim, :dim] *= scale

    return t
