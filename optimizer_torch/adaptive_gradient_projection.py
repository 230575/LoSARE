import torch


class AdaptiveGradientProjector:
    def __init__(
        self,
        rank,
        verbose=False,
        update_proj_gap=200,
        alpha=1.0,
        proj_type="std",
        adaptive_update=True,
        adapt_lr=0.1,
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.alpha = alpha
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.adaptive_update = adaptive_update
        self.adapt_lr = adapt_lr

    def project(self, full_rank_grad, iter):

        device = full_rank_grad.device
        dtype = full_rank_grad.dtype

        def ensure_init(type_str):
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type=type_str
                )
            elif (not self.adaptive_update) and iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type=type_str
                )

        def incremental_update_right(G):
            if self.ortho_matrix is None:
                return
            V = self.ortho_matrix.t().to(device=device, dtype=torch.float)
            Gf = G.to(dtype=torch.float)
            T = Gf @ V
            Y = Gf.transpose(0, 1) @ T
            try:
                Q, _ = torch.linalg.qr(Y, mode="reduced")
            except RuntimeError:
                U, _, _ = torch.linalg.svd(Y, full_matrices=False)
                Q = U[:, : self.rank]
            V_new = (1.0 - self.adapt_lr) * V + self.adapt_lr * Q
            Q2, _ = torch.linalg.qr(V_new, mode="reduced")
            self.ortho_matrix = Q2.t().to(dtype=dtype)

        def incremental_update_left(G):
            if self.ortho_matrix is None:
                return
            U = self.ortho_matrix.to(device=device, dtype=torch.float)
            Gf = G.to(dtype=torch.float)
            T = Gf.transpose(0, 1) @ U
            Y = Gf @ T
            try:
                Q, _ = torch.linalg.qr(Y, mode="reduced")
            except RuntimeError:
                U_s, _, _ = torch.linalg.svd(Y, full_matrices=False)
                Q = U_s[:, : self.rank]
            U_new = (1.0 - self.adapt_lr) * U + self.adapt_lr * Q
            Q2, _ = torch.linalg.qr(U_new, mode="reduced")
            self.ortho_matrix = Q2.to(dtype=dtype)

        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                ensure_init("right")
                if self.adaptive_update:
                    incremental_update_right(full_rank_grad)
                low_rank_grad = torch.matmul(
                    full_rank_grad, self.ortho_matrix.t().to(device)
                )
            else:
                ensure_init("left")
                if self.adaptive_update:
                    incremental_update_left(full_rank_grad)
                low_rank_grad = torch.matmul(
                    self.ortho_matrix.t().to(device), full_rank_grad
                )

        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                ensure_init("left")
                if self.adaptive_update:
                    incremental_update_left(full_rank_grad)
                low_rank_grad = torch.matmul(
                    self.ortho_matrix.t().to(device), full_rank_grad
                )
            else:
                ensure_init("right")
                if self.adaptive_update:
                    incremental_update_right(full_rank_grad)
                low_rank_grad = torch.matmul(
                    full_rank_grad, self.ortho_matrix.t().to(device)
                )

        elif self.proj_type == "right":
            ensure_init("right")
            if self.adaptive_update:
                incremental_update_right(full_rank_grad)
            low_rank_grad = torch.matmul(
                full_rank_grad, self.ortho_matrix.t().to(device)
            )

        elif self.proj_type == "left":
            ensure_init("left")
            if self.adaptive_update:
                incremental_update_left(full_rank_grad)
            low_rank_grad = torch.matmul(
                self.ortho_matrix.t().to(device), full_rank_grad
            )

        elif self.proj_type == "full":
            if self.ortho_matrix is None or (
                not self.adaptive_update and iter % self.update_proj_gap == 0
            ):
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full"
                )
            A, B = self.ortho_matrix
            low_rank_grad = (
                torch.matmul(A.t().to(device), full_rank_grad)
                @ B.t().to(device)
            )

        else:
            raise ValueError("proj_type should be one of std, reverse_std, right, left, full")

        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device)
                )
            else:
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device), low_rank_grad
                )

        elif self.proj_type == "reverse_std":
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device), low_rank_grad
                )
            else:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device)
                )

        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(
                low_rank_grad, self.ortho_matrix.to(low_rank_grad.device)
            )

        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(
                self.ortho_matrix.to(low_rank_grad.device), low_rank_grad
            )

        elif self.proj_type == "full":
            full_rank_grad = (
                torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device), low_rank_grad)
                @ self.ortho_matrix[1].to(low_rank_grad.device)
            )

        else:
            raise ValueError("proj_type should be one of std, reverse_std, right, left, full")

        return full_rank_grad * self.alpha

    def get_orthogonal_matrix(self, weights, rank, type):

        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if type == "right":
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B

        elif type == "left":
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A

        elif type == "full":
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]

        else:
            raise ValueError("type should be left, right or full")
