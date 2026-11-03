"""
Candidate models for the Sirius bimanual-interference data, all fit to the same
12,937 trials in the same order as Vera's Models A, B and C.

Vera's models are built with her own scripts (Model_A/B/C/Build_Model_*.py), unchanged,
except "B-wide", which is Build_Model_B.py with the between-session sd of the lapse
logits widened from HalfNormal(0.5) to HalfNormal(1.5).

The new models extend Model C (hierarchical over sessions, one group per hand,
distractor amplitude d as a covariate), one change at a time:

    psi_main = gh + (1 - gh - gl) * invlogit(b0 + b1*x + b2*d [+ b3*c] [+ db1*c*x])

    L  lapse rates per condition: bimanual trials get their own lapse logits
       (own hand-level mean and own session variation), as in Model B
    K  the bimanual lapse means depend on the session's distractor amplitude:
       logit-lapse_bi = mu_bi + kappa * (d_s - mean d) + sig_bi * z
    P  distractor-presence term b3 * c (session-varying), so the bimanual
       offset is b3 + b2*d rather than proportional to d
    S  bimanual slope change db1 (session-varying), so the distractor can change the JND
    M  capture mixture: on a fraction pi (per hand) of bimanual trials the monkey
       judges a different vibration with its unimanual curve:
         'max'  the stronger of the two vibrations
         'dist' the distractor (the "wrong hand")
       P(high | bi) = (1 - pi) * psi_main + pi * psi_uni(x_capture)

x = (stim_amp - 28) / 14.63, d = dist_amp / 14.63 (0 on unimanual trials), c = 1 on
bimanual trials. Priors of the shared parts are Model C's; new parts:
    mu_b3 ~ N(0, 1), sig_b3 ~ HalfNormal(1); mu_db1 ~ N(0, 1), sig_db1 ~ HalfNormal(1);
    bimanual lapse means ~ N(-3, 1.5), sds HalfNormal(0.5) (as Model C's);
    kappa ~ N(0, 2); logit(pi) ~ N(-2.5, 1.5).
"""
import os
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pickle
from pathlib import Path

W = Path(os.environ.get('MP_REPO', Path(__file__).resolve().parents[2]))   # repository root: Model_Comparison/<package>/models.py
X_SIG = float(np.std([6, 12, 18, 24, 32, 38, 44, 50]))
X_MU = 28.0
D_CENTER = X_MU / X_SIG          # x_dist of a distractor at the 28 um boundary

FAMILY = {                        # name -> switches of build_family
    'C':        dict(),
    'C+L':      dict(L=True),
    'C+LP':     dict(L=True, P=True),
    'C+LPS':    dict(L=True, P=True, S=True),
    'C+LK':     dict(L=True, K=True),
    'C+LKP':    dict(L=True, K=True, P=True),
    'C+LKPS':   dict(L=True, K=True, P=True, S=True),
    'C+Mmax':   dict(M='max'),
    'C+Mdist':  dict(M='dist'),
    'C+LMmax':  dict(L=True, M='max'),
    'C+LPMmax': dict(L=True, P=True, M='max'),
}
VERA = ['A', 'B', 'B-wide']
ALL_MODELS = VERA + list(FAMILY)


def load_data():
    dC = pickle.load(open(W / 'Model_C/Data_C/ReadyData_Sirius_C.pkl', 'rb'))
    dB = pickle.load(open(W / 'Model_B/Data_B/ReadyData_Sirius_B.pkl', 'rb'))
    dA = pickle.load(open(W / 'Model_A/Data_A/ReadyData_Sirius_A.pkl', 'rb'))
    for other in (dA, dB):
        assert np.array_equal(other['resp'], dC['resp']) and np.allclose(other['cov_mat'][:, 1], dC['cov_mat'][:, 1])
    assert np.array_equal(dB['sess_idx'], dC['sess_idx'])
    cov = dC['cov_mat']
    return dict(x=cov[:, 1], d=cov[:, 2], c=(cov[:, 2] > 0).astype(int), hand=dC['grp_idx'],
                sess=dC['sess_idx'], y=dC['resp'].astype(int), sess_d=dC['sess_x_dist'],
                sessions=list(dC['dates_sess_idx']), grp4=dB['grp_idx'],
                dA=dA, dB=dB, dC=dC)


def build_vera(name, data):
    """Vera's Models A, B, C from her own build scripts; 'B-wide' widens B's lapse-sd prior."""
    dA, dB, dC = data['dA'], data['dB'], data['dC']
    if name == 'A':
        ns = {'cov_mat': dA['cov_mat'], 'grp_idx': dA['grp_idx'], 'obs_data': dA['resp']}
        exec((W / 'Model_A/Build_Model_A.py').read_text(), ns)
        return ns['model_A']
    if name in ('B', 'B-wide'):
        src = (W / 'Model_B/Build_Model_B.py').read_text()
        if name == 'B-wide':
            old = "sig_gams = pm.Deterministic('sig_gams', pm.math.abs(0.5*z_sig_gams)"
            assert src.count(old) == 1
            src = src.replace(old, "sig_gams = pm.Deterministic('sig_gams', pm.math.abs(1.5*z_sig_gams)")
        ns = {'cov_mat': dB['cov_mat'], 'grp_idx': dB['grp_idx'], 'obs_data': dB['resp'],
              'sess_idx': dB['sess_idx'], 'sessions': list(dB['dates_sess_idx'])}
        exec(src, ns)
        return ns['model_B']
    if name == 'C-vera':
        ns = {'cov_mat': dC['cov_mat'], 'grp_idx': dC['grp_idx'], 'obs_data': dC['resp'], 'sess_idx': dC['sess_idx'],
              'sess_x_dist': dC['sess_x_dist'], 'sessions': list(dC['dates_sess_idx'])}
        exec((W / 'Model_C/Build_Model_C.py').read_text(), ns)
        return ns['model_C']
    raise ValueError(name)


def build_family(data, L=False, K=False, P=False, S=False, M=None, sig_gam_scale=0.5):
    assert not (K and not L)
    sessions = data['sessions']
    d_bar = float(np.mean(data['sess_d']))
    coords = {'groups': ['left', 'right'], 'betas': ['b0', 'b1', 'b2'], 'lapses': ['gam_h', 'gam_l'],
              'sessions': sessions, 'trials': np.arange(len(data['y']))}
    with pm.Model(coords=coords) as model:
        x = pm.Data('x', data['x'], dims='trials')
        d = pm.Data('d', data['d'], dims='trials')
        c = pm.Data('c', data['c'], dims='trials')
        h = pm.Data('hand', data['hand'], dims='trials')
        s = pm.Data('sess', data['sess'], dims='trials')
        sess_d = pm.Data('sess_d', data['sess_d'], dims='sessions')
        obs = pm.Data('obs', data['y'], dims='trials')

        # b0, b1, b2: exactly Model C's parameterization and priors
        z_mu_b0 = pm.Normal('z_mu_b0', 0, 1, dims='groups')
        z_mu_b1 = pm.Normal('z_mu_b1', 0, 1, dims='groups')
        z_mu_b2 = pm.Normal('z_mu_b2', 0, 1, dims='groups')
        mu_betas = pm.Deterministic('mu_betas', pt.stack([2 * z_mu_b0, pt.exp(np.log(4) + 0.5 * z_mu_b1), z_mu_b2]),
                                    dims=('betas', 'groups'))
        z_sig_b0 = pm.HalfNormal('z_sig_b0', 1, dims='groups')
        u_sig_b1 = pm.Uniform('u_sig_b1', 1e-9, 1 - 1e-9, dims='groups')
        z_sig_b2 = pm.HalfNormal('z_sig_b2', 1, dims='groups')
        sig_betas = pm.Deterministic('sig_betas', pt.stack([2 * z_sig_b0, -pt.log(u_sig_b1), z_sig_b2]),
                                     dims=('betas', 'groups'))
        z_betas = pm.Normal('z_betas', 0, 1, dims=('betas', 'groups', 'sessions'))
        beta_vec = pm.Deterministic('beta_vec', sig_betas[..., None] * z_betas + mu_betas[..., None],
                                    dims=('betas', 'groups', 'sessions'))
        eta = beta_vec[0][h, s] + beta_vec[1][h, s] * x + beta_vec[2][h, s] * d

        if P:   # distractor presence
            mu_b3 = pm.Normal('mu_b3', 0, 1, dims='groups')
            sig_b3 = pm.HalfNormal('sig_b3', 1, dims='groups')
            z_b3 = pm.Normal('z_b3', 0, 1, dims=('groups', 'sessions'))
            b3 = pm.Deterministic('b3', mu_b3[:, None] + sig_b3[:, None] * z_b3, dims=('groups', 'sessions'))
            eta = eta + b3[h, s] * c
        if S:   # bimanual slope change
            mu_db1 = pm.Normal('mu_db1', 0, 1, dims='groups')
            sig_db1 = pm.HalfNormal('sig_db1', 1, dims='groups')
            z_db1 = pm.Normal('z_db1', 0, 1, dims=('groups', 'sessions'))
            db1 = pm.Deterministic('db1', mu_db1[:, None] + sig_db1[:, None] * z_db1, dims=('groups', 'sessions'))
            eta = eta + db1[h, s] * c * x

        # lapse rates: Model C's (shared by both conditions unless L)
        z_mu_gams = pm.Normal('z_mu_gams', 0, 1, dims=('lapses', 'groups'))
        mu_gams = pm.Deterministic('mu_gams', 1.5 * z_mu_gams - 3, dims=('lapses', 'groups'))
        z_sig_gams = pm.HalfNormal('z_sig_gams', 1, dims=('lapses', 'groups'))
        sig_gams = pm.Deterministic('sig_gams', sig_gam_scale * z_sig_gams, dims=('lapses', 'groups'))
        z_gams = pm.Normal('z_gams', 0, 1, dims=('lapses', 'groups', 'sessions'))
        gams = pm.Deterministic('gams', 0.25 * pm.math.invlogit(sig_gams[..., None] * z_gams + mu_gams[..., None]),
                                dims=('lapses', 'groups', 'sessions'))
        gh_u, gl_u = gams[0][h, s], gams[1][h, s]
        if L:
            mu_gams_bi = pm.Normal('mu_gams_bi', -3, 1.5, dims=('lapses', 'groups'))
            sig_gams_bi = pm.HalfNormal('sig_gams_bi', sig_gam_scale, dims=('lapses', 'groups'))
            z_gams_bi = pm.Normal('z_gams_bi', 0, 1, dims=('lapses', 'groups', 'sessions'))
            lin = sig_gams_bi[..., None] * z_gams_bi + mu_gams_bi[..., None]
            if K:
                kappa = pm.Normal('kappa', 0, 2, dims=('lapses', 'groups'))
                lin = lin + kappa[..., None] * (sess_d - d_bar)[None, None, :]
            gams_bi = pm.Deterministic('gams_bi', 0.25 * pm.math.invlogit(lin), dims=('lapses', 'groups', 'sessions'))
            gh = pt.switch(pt.eq(c, 1), gams_bi[0][h, s], gh_u)
            gl = pt.switch(pt.eq(c, 1), gams_bi[1][h, s], gl_u)
        else:
            gh, gl = gh_u, gl_u
        p = gh + (1 - gh - gl) * pm.math.invlogit(eta)

        if M:   # capture mixture on bimanual trials
            logit_pi = pm.Normal('logit_pi', -2.5, 1.5, dims='groups')
            pi = pm.Deterministic('pi', pm.math.invlogit(logit_pi), dims='groups')
            d_stim_scale = d - D_CENTER                      # distractor on the stimulus scale
            x_cap = pt.maximum(x, d_stim_scale) if M == 'max' else d_stim_scale
            p_cap = gh_u + (1 - gh_u - gl_u) * pm.math.invlogit(beta_vec[0][h, s] + beta_vec[1][h, s] * x_cap)
            p = (1 - c * pi[h]) * p + c * pi[h] * p_cap

        pm.Bernoulli('resp', p=pt.clip(p, 1e-8, 1 - 1e-8), observed=obs, dims='trials')
    return model


def build(name, data):
    if name in VERA:
        return build_vera(name, data)
    return build_family(data, **FAMILY[name])


# ---------------------------------------------------------------------------------------
# numpy mirror: P(high) for every trial from posterior draws, for every model
# ---------------------------------------------------------------------------------------
def _invlogit(z):
    return 1 / (1 + np.exp(-z))


def trial_p(name, post, data, draws=None):
    """P(response = 'high') for every trial; post is an xarray posterior (chain, draw, ...).
    Returns (n_draws, n_trials). draws: optional indices into the stacked (chain*draw) axis."""
    st = post.stack(sample=('chain', 'draw'))
    if draws is not None:
        st = st.isel(sample=draws)
    def v(var):  # sample axis first
        a = st[var]
        return np.moveaxis(a.values, -1, 0)
    x, d, c, h, s = data['x'], data['d'], data['c'], data['hand'], data['sess']
    if name == 'A':
        g = data['dA']['grp_idx']
        bv = v('beta_vec')   # (S, betas, groups)
        gh, gl = v('gam_h')[:, g], v('gam_l')[:, g]
        return gh + (1 - gh - gl) * _invlogit(bv[:, 0, g] + bv[:, 1, g] * x)
    if name in ('B', 'B-wide'):
        g = data['grp4']
        bv = v('beta_vec')   # (S, betas, groups, sessions)
        gh, gl = v('gam_h')[:, g, s], v('gam_l')[:, g, s]
        return gh + (1 - gh - gl) * _invlogit(bv[:, 0, g, s] + bv[:, 1, g, s] * x)
    cfg = FAMILY[name] if name in FAMILY else {}
    bv = v('beta_vec')       # (S, 3, groups, sessions)
    eta = bv[:, 0, h, s] + bv[:, 1, h, s] * x + bv[:, 2, h, s] * d
    if cfg.get('P'):
        eta = eta + v('b3')[:, h, s] * c
    if cfg.get('S'):
        eta = eta + v('db1')[:, h, s] * c * x
    gams = v('gams')         # (S, 2, groups, sessions)
    gh_u, gl_u = gams[:, 0, h, s], gams[:, 1, h, s]
    if cfg.get('L'):
        gb = v('gams_bi')
        gh = np.where(c == 1, gb[:, 0, h, s], gh_u)
        gl = np.where(c == 1, gb[:, 1, h, s], gl_u)
    else:
        gh, gl = gh_u, gl_u
    p = gh + (1 - gh - gl) * _invlogit(eta)
    if cfg.get('M'):
        pi = v('pi')[:, h]
        dsc = d - D_CENTER
        x_cap = np.maximum(x, dsc) if cfg['M'] == 'max' else dsc
        p_cap = gh_u + (1 - gh_u - gl_u) * _invlogit(bv[:, 0, h, s] + bv[:, 1, h, s] * x_cap)
        p = (1 - c * pi) * p + c * pi * p_cap
    return p
