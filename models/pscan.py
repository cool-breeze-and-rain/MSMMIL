import math

import torch
import torch.nn.functional as F

"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""


def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


def pad_npo2_2d(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, H, W, D, N)

    Returns:
        Y : (B, npo2(H), npo2(W), D, N)
    """
    len_npo2_h = npo2(X.size(1))
    len_npo2_w = npo2(X.size(2))
    # pad_tuple is reversed
    pad_tuple = (
        0, 0,  # no padding for N
        0, 0,  # no padding for D
        0, len_npo2_w - X.size(2),  # padding for W, all padding is from the right
        0, len_npo2_h - X.size(1)  # padding for H, all padding is from the right
    )
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # �A : (B, D, L, N)
        # �X : (B, D, L, N)

        # �modifies X in place by doing a parallel scan.
        # �more formally, X will be populated by these values :
        # �H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # �which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
        # �only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # �up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X  # X: input array whose some values are replaced; Xa: points to the rightmost value of X
        for i in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)  # split by half to aggregate
            Xa = Xa.view(B, D, T // 2, 2, -1)  # split by half to aggregate

            # BX += A.BX
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))  # X_right_side += A_right_side * X_left_side
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])  # A_right_side += A_left_side

            Aa = Aa[:, :, :, 1]  # ignore left side, keep the right side that stores aggregation result
            Xa = Xa[:, :, :, 1]  # ignore left side, keep the right side that stores aggregation result

        # �we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # �down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):  # same number of steps of up sweep
            Aa = A[:, :, 2 ** k - 1:L:2 ** k]
            Xa = X[:, :, 2 ** k - 1:L:2 ** k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # �A : (B, D, L, N)
        # �X : (B, D, L, N)

        # �the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # �it is used in the backward pass

        # �only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # �up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # �we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # �down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 0:L:2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2 ** k]
            Xa = X[:, :, 0:L:2 ** k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # �cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # �pad tensors (and clone btw)
            A = pad_npo2(A_in)  # �(B, npo2(L), D, N)
            X = pad_npo2(X_in)  # �(B, npo2(L), D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # �(B, D, npo2(L), N)
        X = X.transpose(2, 1)  # �(B, D, npo2(L), N)

        # �parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        # �slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # �the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in)  # �(B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # �(B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # �(B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:],
                                    (0, 0, 0, 1))  # �(B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # �reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


class PScan_2D(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # �A : (BS, D, H, W, N)
        # �X : (BS, D, H, W, N)

        # �modifies X in place by doing a parallel scan.
        # �more formally, X will be populated by these values :
        # �H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # �which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
        # �only supports L that is a power of two (mainly for a clearer code)

        BS, D, H, W, _ = A.size()
        num_steps = int(math.log2(W))

        #  ---------- STAGE 1: HORIZONTAL PARALLEL SCAN  -----------------
        # �up sweep (last 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        A_raw = A.clone()
        Aa = A
        Xa = X  # X: input array whose some values are replaced; Xa: points to the rightmost value of X
        for _ in range(num_steps - 2):
            T = Xa.size(3)
            Aa = Aa.view(BS, D, H, T // 2, 2, -1)  # split by half to aggregate horizontally
            Xa = Xa.view(BS, D, H, T // 2, 2, -1)  # split by half to aggregate horizontally

            Xa[:, :, :, :, 1].add_(
                Aa[:, :, :, :, 1].mul(Xa[:, :, :, :, 0]))  # X_right_side += A_right_side * X_left_side
            Aa[:, :, :, :, 1].mul_(Aa[:, :, :, :, 0])  # A_right_side += A_left_side

            Aa = Aa[:, :, :, :, 1]  # ignore left side, keep the right side that stores aggregation result
            Xa = Xa[:, :, :, :, 1]  # ignore left side, keep the right side that stores aggregation result

        # �we have only 4, 2 or 1 nodes left
        # �A : (BS, D, H, W, N)
        down_sweep_horizontal = True
        if Xa.size(3) == 4:
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Xa[:, :, :, 3].add_(Aa[:, :, :, 3].mul(Xa[:, :, :, 2] + Aa[:, :, :, 2].mul(Xa[:, :, :, 1])))
        elif Xa.size(3) == 2:
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            down_sweep_horizontal = False
        else:
            down_sweep_horizontal = False

        # �down sweep (first 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        if down_sweep_horizontal:
            Aa = A[:, :, :, 2 ** (num_steps - 2) - 1:W:2 ** (num_steps - 2)]
            Xa = X[:, :, :, 2 ** (num_steps - 2) - 1:W:2 ** (num_steps - 2)]
            Xa[:, :, :, 2].add_(Aa[:, :, :, 2].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 2].mul_(Aa[:, :, :, 1])

            for k in range(num_steps - 3, -1, -1):  # same number of steps of up sweep
                Aa = A[:, :, :, 2 ** k - 1:W:2 ** k]
                Xa = X[:, :, :, 2 ** k - 1:W:2 ** k]

                T = Xa.size(3)
                Aa = Aa.view(BS, D, H, T // 2, 2, -1)
                Xa = Xa.view(BS, D, H, T // 2, 2, -1)

                Xa[:, :, :, 1:, 0].add_(Aa[:, :, :, 1:, 0].mul(Xa[:, :, :, :-1, 1]))
                Aa[:, :, :, 1:, 0].mul_(Aa[:, :, :, :-1, 1])

        #  ---------- STAGE 2: VERTICAL PARALLEL SCAN  -----------------
        # �up sweep (last 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        Aa = A_raw
        Xa = X  # X: input array whose some values are replaced; Xa: points to the rightmost value of X

        num_steps = int(math.log2(H))
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(BS, D, T // 2, 2, W, -1)  # split by half to aggregate horizontally
            Xa = Xa.view(BS, D, T // 2, 2, W, -1)  # split by half to aggregate horizontally

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))  # X_right_side += A_right_side * X_left_side
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])  # A_right_side += A_left_side

            Aa = Aa[:, :, :, 1]  # ignore left side, keep the right side that stores aggregation result
            Xa = Xa[:, :, :, 1]  # ignore left side, keep the right side that stores aggregation result

        # �we have only 4, 2 or 1 nodes left
        down_sweep_vertical = True
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            down_sweep_vertical = False
        else:
            down_sweep_vertical = False

        # �down sweep (first 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        if down_sweep_vertical:
            Aa = A[:, :, 2 ** (num_steps - 2) - 1:H:2 ** (num_steps - 2)]
            Xa = X[:, :, 2 ** (num_steps - 2) - 1:H:2 ** (num_steps - 2)]
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
            Aa[:, :, 2].mul_(Aa[:, :, 1])

            for k in range(num_steps - 3, -1, -1):  # same number of steps of up sweep
                Aa = A[:, :, 2 ** k - 1:H:2 ** k]
                Xa = X[:, :, 2 ** k - 1:H:2 ** k]

                T = Xa.size(2)
                Aa = Aa.view(BS, D, T // 2, 2, W, -1)
                Xa = Xa.view(BS, D, T // 2, 2, W, -1)

                Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
                Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # �A : (B, D, H, W, N)
        # �X : (B, D, H, W, N)

        # �the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # �it is used in the backward pass

        # �only supports L that is a power of two (mainly for a clearer code)

        BS, D, H, W, _ = A.size()
        num_steps = int(math.log2(W))

        #  ---------- STAGE 1: HORIZONTAL PARALLEL SCAN  -----------------
        # �up sweep (last 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        A_raw = A.clone()
        Aa = A
        Xa = X  # X: input array whose some values are replaced; Xa: points to the rightmost value of X
        for _ in range(num_steps - 2):
            T = Xa.size(3)
            Aa = Aa.view(BS, D, H, T // 2, 2, -1)  # split by half to aggregate horizontally
            Xa = Xa.view(BS, D, H, T // 2, 2, -1)  # split by half to aggregate horizontally

            Xa[:, :, :, :, 0].add_(
                Aa[:, :, :, :, 0].mul(Xa[:, :, :, :, 1]))  # X_right_side += A_right_side * X_left_side
            Aa[:, :, :, :, 0].mul_(Aa[:, :, :, :, 1])  # A_right_side += A_left_side

            Aa = Aa[:, :, :, :, 0]  # ignore left side, keep the right side that stores aggregation result
            Xa = Xa[:, :, :, :, 0]  # ignore left side, keep the right side that stores aggregation result

        # �we have only 4, 2 or 1 nodes left
        # �A : (BS, D, H, W, N)
        down_sweep_horizontal = True
        if Xa.size(3) == 4:
            Xa[:, :, :, 2].add_(Aa[:, :, :, 2].mul(Xa[:, :, :, 3]))
            Aa[:, :, :, 2].mul_(Aa[:, :, :, 3])
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1] + Aa[:, :, :, 1].mul(Xa[:, :, :, 2])))
        elif Xa.size(3) == 2:
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            down_sweep_horizontal = False
        else:
            down_sweep_horizontal = False

        # �down sweep (first 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        if down_sweep_horizontal:
            Aa = A[:, :, :, 0:W:2 ** (num_steps - 2)]
            Xa = X[:, :, :, 0:W:2 ** (num_steps - 2)]
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 2]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 2])

            for k in range(num_steps - 3, -1, -1):  # same number of steps of up sweep
                Aa = A[:, :, :, 0:W:2 ** k]
                Xa = X[:, :, :, 0:W:2 ** k]

                T = Xa.size(3)
                Aa = Aa.view(BS, D, H, T // 2, 2, -1)
                Xa = Xa.view(BS, D, H, T // 2, 2, -1)

                Xa[:, :, :, :-1, 1].add_(Aa[:, :, :, :-1, 1].mul(Xa[:, :, :, 1:, 1]))
                Aa[:, :, :, :-1, 1].mul_(Aa[:, :, :, 1:, 0])

        #  ---------- STAGE 2: VERTICAL PARALLEL SCAN  -----------------
        # �up sweep (last 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        Aa = A_raw
        Xa = X  # X: input array whose some values are replaced; Xa: points to the rightmost value of X

        num_steps = int(math.log2(H))
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(BS, D, T // 2, 2, W, -1)  # split by half to aggregate horizontally
            Xa = Xa.view(BS, D, T // 2, 2, W, -1)  # split by half to aggregate horizontally

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))  # X_right_side += A_right_side * X_left_side
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])  # A_right_side += A_left_side

            Aa = Aa[:, :, :, 0]  # ignore left side, keep the right side that stores aggregation result
            Xa = Xa[:, :, :, 0]  # ignore left side, keep the right side that stores aggregation result

        # �we have only 4, 2 or 1 nodes left
        down_sweep_vertical = True
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1] + Aa[:, :, 1].mul(Xa[:, :, 2])))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            down_sweep_vertical = False
        else:
            down_sweep_vertical = False

        # �down sweep (first 2 steps unfolded)
        # �A : (BS, D, H, W, N)
        if down_sweep_vertical:
            Aa = A[:, :, 0:H:2 ** (num_steps - 2)]
            Xa = X[:, :, 0:H:2 ** (num_steps - 2)]
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
            Aa[:, :, 1].mul_(Aa[:, :, 2])

            for k in range(num_steps - 3, -1, -1):  # same number of steps of up sweep
                Aa = A[:, :, 0:H:2 ** k]
                Xa = X[:, :, 0:H:2 ** k]

                T = Xa.size(2)
                Aa = Aa.view(BS, D, T // 2, 2, W, -1)
                Xa = Xa.view(BS, D, T // 2, 2, W, -1)

                Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 0].mul(Xa[:, :, 1:, 1]))
                Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (BS, H, W, D, N)
            X_in : (BS, H, W, D, N)

        Returns:
            H : (B, H, W, D, N)
        """

        H, W = X_in.size(1), X_in.size(2)

        # �cloning is required because of the in-place ops
        if W == npo2(W) and H == npo2(H):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2_2d(A_in)  # �(B, npo2(L), D, N)
            X = pad_npo2_2d(X_in)  # �(B, npo2(L), D, N)

        # prepare tensors
        A = A.permute(0, 3, 1, 2, 4)  # �(BS, D, H, W, N)
        X = X.permute(0, 3, 1, 2, 4)  # �(BS, D, H, W, N)

        # �parallel scan (modifies A and X in-place)

        PScan_2D.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        X = X[:, :, :H, :W, :]
        return X.permute(0, 2, 3, 1, 4)  # (BS, D, H, W, N) -> (BS, H, W, D, N)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (BS, H, W, D, N), X : (BS, D, H, W, N)
            grad_output_in : (B, H, W, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        _, H, W, _, _ = grad_output_in.size()

        # cloning is requiered because of the in-place ops
        if H == npo2(H) and W == npo2(W):
            grad_output = grad_output_in.clone()
            # �the next padding will clone A_in
        else:
            # raise NotImplementedError()
            grad_output = pad_npo2_2d(grad_output_in)  # �(B, npo2(L), D, N)
            A_in = pad_npo2_2d(A_in)  # �(B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.permute(0, 3, 1, 2, 4)  # (BS, D, H, W, N)
        A_in = A_in.permute(0, 3, 1, 2, 4)  # �(B, D, npo2(H), npo2(W), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:, 1:],
                                    (0, 0, 0, 1, 0, 1))  # �shift 1 to the left (see hand derivation) # TODO: check why

        # �reverse parallel scan (modifies grad_output in-place)
        PScan_2D.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)  # (BS, D, H, W, N)
        Q[:, :, 1:, 1:].add_(X[:, :, :-1, :-1] * grad_output[:, :, 1:, 1:])

        return Q.permute(0, 2, 3, 1, 4)[:, :H, :W], grad_output.permute(0, 2, 3, 1, 4)[:, :H, :W]


pscan = PScan.apply
pscan_2d = PScan_2D.apply