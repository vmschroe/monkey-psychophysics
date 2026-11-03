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
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
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

    def run(spec, cells, cell_fn):
        """spec: list of (name, 'r', pop_m (J,), pop_s (J,), post_u (S,)) or (name, 'f', value (J,));
        returns log p(y | phi_j) for every j and the IS ESS."""
        xs, cs, n, k = cells
        rand = [t for t in spec if t[1] == 'r']
        fixed = {t[0]: t[2] for t in spec if t[1] == 'f'}
        col = {t[0]: i for i, t in enumerate(rand)}
        def cell_p(u, jj):
            par = {nm: u[..., i:i + 1] for nm, i in col.items()}
            for nm, v in fixed.items():
                par[nm] = v.mean() if jj is None else v[jj][:, None, None]
            return cell_fn(par, xs, cs)
        if not rand:
            J = len(next(iter(fixed.values())))
            return binom_ll(cell_p(np.zeros((J, 1, 0)), np.arange(J)), n, k)[:, 0], np.full(J, np.nan)
        pop_m = np.stack([t[2] for t in rand], 1); pop_s = np.stack([t[3] for t in rand], 1) + 1e-6
        post_u = np.stack([t[4] for t in rand], 1)
        return integrated_loglik(post_u, pop_m, pop_s, cell_p, n, k, rng)

    if name in ('B', 'B-wide', 'B-fixlapse'):
        g4 = data['grp4']
        mb, sb, bvf = arr(ph, 'mu_betas'), arr(ph, 'sig_betas'), arr(full, 'beta_vec')   # (J,2,4) ...
        mg = arr(ph, 'mu_gams')                                                           # (J,2,4): b0->gam_h, b1->gam_l
        fixl = name == 'B-fixlapse'
        if not fixl:
            sg, mgf, sgf, zgf = arr(ph, 'sig_gams'), arr(full, 'mu_gams'), arr(full, 'sig_gams'), arr(full, 'z_gams')
        def cell_fn(par, xs, cs):
            gh_ = 0.25 * expit(par['lh']); gl_ = 0.25 * expit(par['ll'])
            return gh_ + (1 - gh_ - gl_) * expit(par['b0'] + par['b1'] * xs)
        for s in sessions_to_do:
            for g in range(4):
                cells = cells_for(data, (sess == s) & (g4 == g))
                spec = [('b0', 'r', mb[:, 0, g], sb[:, 0, g], bvf[:, 0, g, s]),
                        ('b1', 'r', mb[:, 1, g], sb[:, 1, g], bvf[:, 1, g, s])]
                for kk, nm in enumerate(['lh', 'll']):
                    spec.append((nm, 'f', mg[:, kk, g]) if fixl else
                                (nm, 'r', mg[:, kk, g], sg[:, kk, g], mgf[:, kk, g] + sgf[:, kk, g] * zgf[:, kk, g, s]))
                lp, ess = run(spec, cells, cell_fn)
                logp[:, s] += lp
                ess_min[s] = np.nanmin([ess_min[s], np.nanmin(ess) if np.isfinite(ess).any() else np.inf])
        return logp, ess_min

    cfg = FAMILY[name]
    L_, K_, P_, S_, M_ = (cfg.get(key) for key in ('L', 'K', 'P', 'S', 'M'))
    SH_, CR_, OR_ = cfg.get('shared_b0'), cfg.get('crit'), cfg.get('order')
    rnd = {nm: cfg.get(f'sess_{nm}', True) for nm in ('b0', 'b1', 'b2', 'b3', 'lapse')}
    d_bar = float(np.mean(data['sess_d']))
    mb, sb, bvf = arr(ph, 'mu_betas'), arr(ph, 'sig_betas'), arr(full, 'beta_vec')     # (J,3,2), (S,3,2,42)
    mg = arr(ph, 'mu_gams')
    if rnd['lapse']:
        sg, mgf, sgf, zgf = arr(ph, 'sig_gams'), arr(full, 'mu_gams'), arr(full, 'sig_gams'), arr(full, 'z_gams')
    if P_:
        m3, b3f = arr(ph, 'mu_b3'), arr(full, 'b3')
        if rnd['b3']:
            s3 = arr(ph, 'sig_b3')
    if S_:
        mdb, sdb, dbf = arr(ph, 'mu_db1'), arr(ph, 'sig_db1'), arr(full, 'db1')
    if L_:
        mgb = arr(ph, 'mu_gams_bi')
        if rnd['lapse']:
            sgb, mgbf, sgbf, zgbf = arr(ph, 'sig_gams_bi'), arr(full, 'mu_gams_bi'), arr(full, 'sig_gams_bi'), arr(full, 'z_gams_bi')
        if K_:
            kap, kapf = arr(ph, 'kappa'), arr(full, 'kappa')
    if M_:
        pi = arr(ph, 'pi')                                                                  # (J,2)
    if CR_:
        lam = arr(ph, 'lam')                                                                # (J,2)
    if OR_:
        tau = arr(ph, 'tau')
        t_s = (np.arange(nS) - (nS - 1) / 2) / 10
    if SH_:
        ssh, vshf = arr(ph, 'sig_shared'), arr(full, 'sig_shared')[:, None] * arr(full, 'z_shared')   # (J,), (S,42)

    def hand_spec(s, h):
        ds = data['sess_d'][s]
        spec = []
        for kk, nm in enumerate(['b0', 'b1', 'b2']):
            if rnd[nm]:
                m_ = (mb[:, kk, h] + (lam[:, h] * (ds - d_bar) if (CR_ and nm == 'b0') else 0.0)
                      + (tau[:, h] * t_s[s] if (OR_ and nm == 'b0') else 0.0))
                u_ = bvf[:, kk, h, s] - (vshf[:, s] if (SH_ and nm == 'b0') else 0.0)   # hand-specific part
                spec.append((nm, 'r', m_, sb[:, kk, h], u_))
            else:
                spec.append((nm, 'f', mb[:, kk, h] + (lam[:, h] * (ds - d_bar) if (CR_ and nm == 'b0') else 0.0)
                             + (tau[:, h] * t_s[s] if (OR_ and nm == 'b0') else 0.0)))
        if P_:
            spec.append(('b3', 'r', m3[:, h], s3[:, h], b3f[:, h, s]) if rnd['b3'] else ('b3', 'f', m3[:, h]))
        if S_:
            spec.append(('db1', 'r', mdb[:, h], sdb[:, h], dbf[:, h, s]))
        for kk, nm in enumerate(['lh', 'll']):
            spec.append((nm, 'r', mg[:, kk, h], sg[:, kk, h], mgf[:, kk, h] + sgf[:, kk, h] * zgf[:, kk, h, s])
                        if rnd['lapse'] else (nm, 'f', mg[:, kk, h]))
        if L_:
            for kk, nm in enumerate(['lh_bi', 'll_bi']):
                shift = kap[:, kk, h] * (ds - d_bar) if K_ else 0.0
                if rnd['lapse']:
                    shift_f = kapf[:, kk, h] * (ds - d_bar) if K_ else 0.0
                    spec.append((nm, 'r', mgb[:, kk, h] + shift, sgb[:, kk, h],
                                 mgbf[:, kk, h] + shift_f + sgbf[:, kk, h] * zgbf[:, kk, h, s]))
                else:
                    spec.append((nm, 'f', mgb[:, kk, h] + shift))
        if M_:
            spec.append(('pi', 'f', pi[:, h]))
        xs, cs, n, k = cells_for(data, (sess == s) & (hand == h))
        dvec = cs * ds
        x_cap = (np.maximum(xs, ds - D_CENTER) if M_ == 'max' else np.full_like(xs, ds - D_CENTER)) if M_ else None

        def cell_fn(par, xs_, cs_, dvec=dvec, x_cap=x_cap):
            b0 = par['b0'] + (par['v'] if SH_ else 0.0)
            eta = b0 + par['b1'] * xs_ + par['b2'] * dvec
            if P_:
                eta = eta + par['b3'] * cs_
            if S_:
                eta = eta + par['db1'] * cs_ * xs_
            gh_u = 0.25 * expit(par['lh']); gl_u = 0.25 * expit(par['ll'])
            if L_:
                gh = np.where(cs_ == 1, 0.25 * expit(par['lh_bi']), gh_u)
                gl = np.where(cs_ == 1, 0.25 * expit(par['ll_bi']), gl_u)
            else:
                gh, gl = gh_u, gl_u
            p = gh + (1 - gh - gl) * expit(eta)
            if M_:
                p_cap = gh_u + (1 - gh_u - gl_u) * expit(b0 + par['b1'] * x_cap)
                p = (1 - cs_ * par['pi']) * p + cs_ * par['pi'] * p_cap
            return p
        return spec, (xs, cs, n, k), cell_fn

    for s in sessions_to_do:
        if not SH_:
            for h in range(2):
                spec, cells, cell_fn = hand_spec(s, h)
                lp, ess = run(spec, cells, cell_fn)
                logp[:, s] += lp
                ess_min[s] = np.nanmin([ess_min[s], np.nanmin(ess) if np.isfinite(ess).any() else np.inf])
        else:
            # the shared daily offset v couples the hands: integrate both hands and v jointly
            parts = [hand_spec(s, h) for h in range(2)]
            spec = [('v', 'r', np.zeros(len(ssh)), ssh, vshf[:, s])]
            for h, (sp, _, _) in enumerate(parts):
                spec += [(f'{t[0]}_{h}',) + t[1:] for t in sp]
            xs = np.concatenate([pt_[1][0] for pt_ in parts]); cs = np.concatenate([pt_[1][1] for pt_ in parts])
            n = np.concatenate([pt_[1][2] for pt_ in parts]); k = np.concatenate([pt_[1][3] for pt_ in parts])
            n0 = len(parts[0][1][0])
            def joint_fn(par, xs_, cs_):
                out = []
                for h, (sp, cells, fn) in enumerate(parts):
                    ph_ = {t[0]: par[f'{t[0]}_{h}'] for t in sp}
                    ph_['v'] = par['v']
                    sl = slice(0, n0) if h == 0 else slice(n0, None)
                    out.append(fn(ph_, xs_[sl], cs_[sl]))
                return np.concatenate(out, axis=-1)
            lp, ess = run(spec, (xs, cs, n, k), joint_fn)
            logp[:, s] += lp
            ess_min[s] = np.nanmin([ess_min[s], np.nanmin(ess) if np.isfinite(ess).any() else np.inf])
    ess_min[~np.isfinite(ess_min)] = np.nan
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
    FITDIR = sys.argv[3] if len(sys.argv) > 3 else str(__import__('pathlib').Path(__file__).resolve().parent / 'fits5000')
    post = az.from_netcdf(f'{FITDIR}/{name}.nc').posterior
    logp, ess_min = loso_model(name, post, data, seed=seed)
    res = psis_loso(logp)
    out = {'model': name, 'seed': seed, 'elpd_loso': float(res.elpd_loo), 'se': float(res.se),
           'max_k': float(res.pareto_k.max()), 'n_k_gt_0.7': int((res.pareto_k > 0.7).sum()),
           'min_is_ess': float(np.nanmin(ess_min)) if np.isfinite(ess_min).any() else None,
           'median_session_min_is_ess': float(np.nanmedian(ess_min)) if np.isfinite(ess_min).any() else None}
    print(json.dumps(out))
    pickle.dump({'logp': logp, 'ess_min': ess_min, 'loo': res, 'summary': out},
                open(f'{FITDIR}/{name}.loso_s{seed}.pkl', 'wb'))
