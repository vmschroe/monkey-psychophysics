"""
Leave-one-session-out (LOSO) cross-validation for every model, without refitting.

For a held-out session s the predictive density is p(y_s | y_-s) = E[p(y_s | phi) | y_-s],
where phi are the hand- (or group-) level parameters and the session's own parameters are
integrated out: p(y_s | phi) = integral p(y_s | u) p(u | phi) du. Steps:
  1. p(y_s | phi_j) for every posterior draw phi_j, by defensive importance sampling over the
     session-level parameters u (proposal: 50% multivariate t fitted to the session's posterior
     draws, 50% the population distribution p(u | phi_j));
  2. PSIS over the phi draws with log-likelihood log p(y_s | phi_j) (Pareto k checks reliability).
The trial likelihood uses counts in (stimulus level x condition) cells of each session and hand.
"""
import sys, pickle, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, arviz as az, xarray as xr
from scipy.special import gammaln, logsumexp, expit
from models import load_data, FAMILY, D_CENTER

N_PHI = 1000       # posterior draws of phi used (thinned)
N_IS = 2000        # importance samples per phi draw
NU = 5             # t proposal degrees of freedom
INFLATE = 1.5      # t proposal scale inflation


def cells_for(data, mask):
    """Counts (n, k) per (x, c) cell among trials in mask. Returns x, c, n, k arrays."""
    x, c, y = data['x'][mask], data['c'][mask], data['y'][mask]
    keys, inv = np.unique(np.c_[np.round(x, 6), c], axis=0, return_inverse=True)
    n = np.bincount(inv.ravel(), minlength=len(keys)); k = np.bincount(inv.ravel(), weights=y, minlength=len(keys))
    return keys[:, 0], keys[:, 1].astype(int), n, k


def binom_ll(p, n, k):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return (k * np.log(p) + (n - k) * np.log1p(-p)).sum(-1)


def log_t(u, mu, L_inv, logdet, nu):
    p = u.shape[-1]
    z = (u - mu) @ L_inv.T
    q = (z ** 2).sum(-1)
    return gammaln((nu + p) / 2) - gammaln(nu / 2) - p / 2 * np.log(nu * np.pi) - 0.5 * logdet - (nu + p) / 2 * np.log1p(q / nu)


def log_norm_diag(u, m, s):
    return (-0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((u - m) / s) ** 2).sum(-1)


def _grad_hess(f, u0, eps=1e-3):
    """Central-difference gradient and Hessian of a scalar function f at u0."""
    D = len(u0); E = np.eye(D) * eps
    g = np.array([(f(u0 + E[a]) - f(u0 - E[a])) / (2 * eps) for a in range(D)])
    H = np.empty((D, D))
    for a in range(D):
        for b in range(a, D):
            H[a, b] = H[b, a] = (f(u0 + E[a] + E[b]) - f(u0 + E[a] - E[b]) - f(u0 - E[a] + E[b])
                                 + f(u0 - E[a] - E[b])) / (4 * eps ** 2)
    return g, H


def integrated_loglik(post_u, pop_m, pop_s, cell_p_fn, n, k, rng):
    """log p(y | phi_j) = log integral p(y | u) N(u; pop_m[j], diag(pop_s[j]^2)) du for every phi draw j.
    post_u: (S, D) posterior draws of the session parameters; cell_p_fn(u, jj) -> (..., n_cells),
    jj = indices of phi draws (None: use average hand-level values).
    Proposal for each j: the session likelihood is approximated by a quadratic in u at the session's
    posterior mean (gradient g, curvature H, clipped to be positive semi-definite) and combined with the
    population N(pop_m[j], pop_s[j]); importance samples come 90% from a t_5 with that mean and 1.2x the
    scale, 10% from the population itself (defensive mixture, so the weights stay bounded).
    Returns log p (J,) and the importance-sampling ESS (J,)."""
    J, D = pop_m.shape
    u0 = post_u.mean(0)
    f = lambda u: binom_ll(cell_p_fn(u[None, None, :], None), n, k).item()
    g, Hn = _grad_hess(f, u0)
    lam, V = np.linalg.eigh(-Hn)
    H = (V * np.clip(lam, 0, None)) @ V.T                                  # curvature of -log L, PSD
    prec = H[None] + np.einsum('jd,de->jde', 1 / pop_s ** 2, np.eye(D))   # (J, D, D)
    rhs = (g + H @ u0)[None] + pop_m / pop_s ** 2
    mean = np.linalg.solve(prec, rhs[..., None])[..., 0]                  # (J, D)
    cov = np.linalg.inv(prec) * 1.2 ** 2
    Lc = np.linalg.cholesky(cov)                                          # (J, D, D)
    Lc_inv = np.linalg.inv(Lc); logdet = 2 * np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)
    out, ess = np.empty(J), np.empty(J)
    n_t = int(0.9 * N_IS); n_p = N_IS - n_t
    for j0 in range(0, J, 100):
        jj = np.arange(j0, min(J, j0 + 100))
        z = rng.standard_normal((len(jj), n_t, D))
        chi = rng.chisquare(NU, size=(len(jj), n_t, 1))
        u_t = mean[jj, None, :] + np.einsum('jde,jne->jnd', Lc[jj], z) / np.sqrt(chi / NU)
        u_p = pop_m[jj, None, :] + pop_s[jj, None, :] * rng.standard_normal((len(jj), n_p, D))
        u = np.concatenate([u_t, u_p], axis=1)                             # (j, N_IS, D)
        lp_pop = log_norm_diag(u, pop_m[jj, None, :], pop_s[jj, None, :])
        zz = np.einsum('jde,jne->jnd', Lc_inv[jj], u - mean[jj, None, :])
        q = (zz ** 2).sum(-1)
        lt = (gammaln((NU + D) / 2) - gammaln(NU / 2) - D / 2 * np.log(NU * np.pi) - 0.5 * logdet[jj, None]
              - (NU + D) / 2 * np.log1p(q / NU))
        lq = np.logaddexp(np.log(0.9) + lt, np.log(0.1) + lp_pop)
        ll = binom_ll(cell_p_fn(u, jj), n, k)                              # (j, N_IS)
        lw = ll + lp_pop - lq
        out[jj] = logsumexp(lw, axis=1) - np.log(N_IS)
        w = np.exp(lw - lw.max(1, keepdims=True))
        ess[jj] = w.sum(1) ** 2 / (w ** 2).sum(1)
    return out, ess


def thin(post, n):
    st = post.stack(sample=('chain', 'draw'))
    idx = np.linspace(0, st.sizes['sample'] - 1, n).astype(int)
    return st.isel(sample=idx), st


def arr(ds, var):  # sample axis first
    return np.moveaxis(ds[var].values, -1, 0)


def loso_model(name, post, data, seed=1, post_u=None, only=None):
    """post: posterior supplying the phi draws; post_u: posterior supplying the session-parameter draws
    used to build the importance-sampling proposal (default: the same); only: sessions to compute."""
    rng = np.random.default_rng(seed)
    ph, full = thin(post, N_PHI)
    if post_u is not None:
        full = post_u.stack(sample=('chain', 'draw'))
    sess, hand = data['sess'], data['hand']
    nS = len(data['sessions'])
    logp = np.zeros((N_PHI, nS))
    ess_min = np.full(nS, np.inf)
    sessions_to_do = range(nS) if only is None else only

    if name == 'A':
        bv, gh, gl = arr(ph, 'beta_vec'), arr(ph, 'gam_h'), arr(ph, 'gam_l')     # (J,2,4), (J,4)
        g4 = data['dA']['grp_idx']
        for s in sessions_to_do:
            for g in range(4):
                xs, cs, n, k = cells_for(data, (sess == s) & (g4 == g))
                p = gh[:, g, None] + (1 - gh[:, g, None] - gl[:, g, None]) * expit(bv[:, 0, g, None] + bv[:, 1, g, None] * xs)
                logp[:, s] += binom_ll(p, n, k)
        ess_min[:] = np.nan
        return logp, ess_min

    if name in ('B', 'B-wide'):
        g4 = data['grp4']
        mb, sb = arr(ph, 'mu_betas'), arr(ph, 'sig_betas')               # (J,2,4)
        mg, sg = arr(ph, 'mu_gams'), arr(ph, 'sig_gams')                 # (J,2,4): b0->gam_h, b1->gam_l
        bvf, mgf, sgf, zgf = arr(full, 'beta_vec'), arr(full, 'mu_gams'), arr(full, 'sig_gams'), arr(full, 'z_gams')
        for s in sessions_to_do:
            for g in range(4):
                xs, cs, n, k = cells_for(data, (sess == s) & (g4 == g))
                pop_m = np.stack([mb[:, 0, g], mb[:, 1, g], mg[:, 0, g], mg[:, 1, g]], 1)
                pop_s = np.stack([sb[:, 0, g], sb[:, 1, g], sg[:, 0, g], sg[:, 1, g]], 1) + 1e-6
                post_u = np.stack([bvf[:, 0, g, s], bvf[:, 1, g, s],
                                   mgf[:, 0, g] + sgf[:, 0, g] * zgf[:, 0, g, s],
                                   mgf[:, 1, g] + sgf[:, 1, g] * zgf[:, 1, g, s]], 1)
                def cell_p(u, jj, xs=xs):
                    gh_ = 0.25 * expit(u[..., 2:3]); gl_ = 0.25 * expit(u[..., 3:4])
                    return gh_ + (1 - gh_ - gl_) * expit(u[..., 0:1] + u[..., 1:2] * xs)
                lp, ess = integrated_loglik(post_u, pop_m, pop_s, cell_p, n, k, rng)
                logp[:, s] += lp
                ess_min[s] = min(ess_min[s], ess.min())
        return logp, ess_min

    cfg = FAMILY[name]
    L_, K_, P_, S_, M_ = (cfg.get(key) for key in ('L', 'K', 'P', 'S', 'M'))
    d_bar = float(np.mean(data['sess_d']))
    mb, sb = arr(ph, 'mu_betas'), arr(ph, 'sig_betas')                   # (J,3,2)
    mg, sg = arr(ph, 'mu_gams'), arr(ph, 'sig_gams')                     # (J,2,2)
    bvf, mgf, sgf, zgf = arr(full, 'beta_vec'), arr(full, 'mu_gams'), arr(full, 'sig_gams'), arr(full, 'z_gams')
    if P_:
        m3, s3, b3f = arr(ph, 'mu_b3'), arr(ph, 'sig_b3'), arr(full, 'b3')
    if S_:
        mdb, sdb, dbf = arr(ph, 'mu_db1'), arr(ph, 'sig_db1'), arr(full, 'db1')
    if L_:
        mgb, sgb = arr(ph, 'mu_gams_bi'), arr(ph, 'sig_gams_bi')
        mgbf, sgbf, zgbf = arr(full, 'mu_gams_bi'), arr(full, 'sig_gams_bi'), arr(full, 'z_gams_bi')
        if K_:
            kap, kapf = arr(ph, 'kappa'), arr(full, 'kappa')
    if M_:
        pi = arr(ph, 'pi')                                               # (J,2)
    for s in sessions_to_do:
        ds = data['sess_d'][s]
        for h in range(2):
            xs, cs, n, k = cells_for(data, (sess == s) & (hand == h))
            cols_m = [mb[:, 0, h], mb[:, 1, h], mb[:, 2, h]]
            cols_s = [sb[:, 0, h], sb[:, 1, h], sb[:, 2, h]]
            cols_u = [bvf[:, 0, h, s], bvf[:, 1, h, s], bvf[:, 2, h, s]]
            if P_:
                cols_m.append(m3[:, h]); cols_s.append(s3[:, h]); cols_u.append(b3f[:, h, s])
            if S_:
                cols_m.append(mdb[:, h]); cols_s.append(sdb[:, h]); cols_u.append(dbf[:, h, s])
            i_lh = len(cols_m)
            for kk in range(2):
                cols_m.append(mg[:, kk, h]); cols_s.append(sg[:, kk, h]); cols_u.append(mgf[:, kk, h] + sgf[:, kk, h] * zgf[:, kk, h, s])
            if L_:
                for kk in range(2):
                    shift = kap[:, kk, h] * (ds - d_bar) if K_ else 0.0
                    shift_f = kapf[:, kk, h] * (ds - d_bar) if K_ else 0.0
                    cols_m.append(mgb[:, kk, h] + shift); cols_s.append(sgb[:, kk, h])
                    cols_u.append(mgbf[:, kk, h] + shift_f + sgbf[:, kk, h] * zgbf[:, kk, h, s])
            pop_m, pop_s, post_u = np.stack(cols_m, 1), np.stack(cols_s, 1) + 1e-6, np.stack(cols_u, 1)
            dvec = cs * ds
            x_cap = (np.maximum(xs, ds - D_CENTER) if M_ == 'max' else np.full_like(xs, ds - D_CENTER)) if M_ else None
            pi_h = pi[:, h] if M_ else None

            def cell_p(u, jj, xs=xs, cs=cs, dvec=dvec, x_cap=x_cap, pi_h=pi_h):
                b0, b1, b2 = u[..., 0:1], u[..., 1:2], u[..., 2:3]
                col = 3
                eta = b0 + b1 * xs + b2 * dvec
                if P_:
                    eta = eta + u[..., col:col + 1] * cs; col += 1
                if S_:
                    eta = eta + u[..., col:col + 1] * cs * xs; col += 1
                gh_u = 0.25 * expit(u[..., i_lh:i_lh + 1]); gl_u = 0.25 * expit(u[..., i_lh + 1:i_lh + 2])
                if L_:
                    gh = np.where(cs == 1, 0.25 * expit(u[..., i_lh + 2:i_lh + 3]), gh_u)
                    gl = np.where(cs == 1, 0.25 * expit(u[..., i_lh + 3:i_lh + 4]), gl_u)
                else:
                    gh, gl = gh_u, gl_u
                p = gh + (1 - gh - gl) * expit(eta)
                if M_:
                    p_cap = gh_u + (1 - gh_u - gl_u) * expit(b0 + b1 * x_cap)
                    pij = pi_h.mean() if jj is None else pi_h[jj][:, None, None]
                    p = (1 - cs * pij) * p + cs * pij * p_cap
                return p
            lp, ess = integrated_loglik(post_u, pop_m, pop_s, cell_p, n, k, rng)
            logp[:, s] += lp
            ess_min[s] = min(ess_min[s], ess.min())
    return logp, ess_min


def psis_loso(logp):
    ll = xr.DataArray(logp[None], dims=('chain', 'draw', 'session'))
    idata = az.InferenceData(posterior=xr.Dataset({'dummy': xr.DataArray(np.zeros((1, logp.shape[0])), dims=('chain', 'draw'))}),
                             log_likelihood=xr.Dataset({'y': ll}))
    return az.loo(idata, pointwise=True, reff=1.0)   # thinned draws: treat as independent


if __name__ == '__main__':
    name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    data = load_data()
    post = az.from_netcdf(f'{Path(__file__).resolve().parent}/fits/{name}.nc').posterior
    logp, ess_min = loso_model(name, post, data, seed=seed)
    res = psis_loso(logp)
    out = {'model': name, 'seed': seed, 'elpd_loso': float(res.elpd_loo), 'se': float(res.se),
           'max_k': float(res.pareto_k.max()), 'n_k_gt_0.7': int((res.pareto_k > 0.7).sum()),
           'min_is_ess': float(np.nanmin(ess_min)) if np.isfinite(ess_min).any() else None,
           'median_session_min_is_ess': float(np.nanmedian(ess_min)) if np.isfinite(ess_min).any() else None}
    print(json.dumps(out))
    pickle.dump({'logp': logp, 'ess_min': ess_min, 'loo': res, 'summary': out},
                open(f'{Path(__file__).resolve().parent}/fits/{name}.loso_s{seed}.pkl', 'wb'))
