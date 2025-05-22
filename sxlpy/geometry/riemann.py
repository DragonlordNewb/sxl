from sympy import Expr
from sxlpy.geometry import MetricTensor
from sxlpy.geometry import Index
from sxlpy.geometry import Manifold
from sxlpy.geometry import Tensor

def compute_riemann_tensor_component(index: Index, m: Manifold) -> Expr:
    g = m.metric
    indices, variances = index.indices, index.variances
    rho, sig, mu, nu = indices
    A = None
    B = None
    if variances = ["u", "d", "d", "d"]:
        A = diff(g.conn_mixed(rho, nu, sig), m.x(mu)) - diff(g.conn_mixed(rho, mu, sig), m.x(nu))
        B = sum(g.conn_mixed(rho,mu,lam)*g.conn_mixed(lam,nu,sig) - g.conn_mixed(rho,nu,lam)*g.con_mixed(lam,mu,sig) for lam in range(dim(m)))
    elif variances = ["d", "d", "d", "d"]:
        A = diff(g.conn_co(rho, nu, sig), m.x(mu)) - diff(g.conn_co(rho, mu, sig), m.x(nu))
        B = sum(g.conn_co(rho,mu,lam)*g.conn_mixed(lam,nu,sig) - g.conn_co(rho,nu,lam)*g.con_mixed(lam,mu,sig) for lam in range(dim(m)))
    if A is not None and B is not None:
        return A + B
    return None

def riemann_tensor() -> Tensor:
    return Tensor