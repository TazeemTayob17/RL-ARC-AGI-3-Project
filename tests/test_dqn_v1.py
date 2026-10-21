import numpy as np

from src.adapter import NO_ACTION, Observation, build_mask
from src.checkpoint_resolver import DEPLOYMENT_FOLD, PUBLIC_GAMES, resolve_prior
from src.config import DQN_V1_DEFAULTS, RunConfig
from src.dqn_v1 import DQNV1
from src.run import build_learner


def _obs(avail=(1,2,3,4,6,7), seed=1):
    r=np.random.default_rng(seed)
    f=r.integers(0,13,(64,64),dtype=np.int8)
    return Observation(f,f.copy(),build_mask(avail),NO_ACTION,0,0,5)


def _learner(game='ls20'):
    spec = resolve_prior(game)
    return DQNV1(dict(DQN_V1_DEFAULTS), prior_path=spec.path, fold=spec.fold,
                 seed_offset=spec.seed_offset, deployment=spec.deployment)


def test_v1_checkpoint_loads_and_action_is_legal():
    d=_learner('ls20')
    d.start_run(5209)
    for avail in ((6,), (1,2,3,4), (3,4,6,7), (5,6,7)):
        o=_obs(avail, 9)
        idx=d.act(o)
        assert o.mask[idx]
        d.pending=None


def test_unknown_game_uses_the_declared_deployment_checkpoint():
    spec = resolve_prior('zz99-0000')
    assert spec.deployment and spec.fold == DEPLOYMENT_FOLD
    d = build_learner(RunConfig(algo='dqn', game='zz99-0000', version='V1'))
    d.start_run(1)
    o = _obs((1, 2, 3, 4), 3)
    assert o.mask[d.act(o)]
    assert d.diagnostics()['deployment_checkpoint'] is True


def test_public_games_keep_their_fold_and_seed_offset():
    assert resolve_prior('ls20').seed_offset == PUBLIC_GAMES.index('ls20') * 9173
    assert not resolve_prior('ls20').deployment
    assert resolve_prior('ls20-9607627b').seed_offset == resolve_prior('ls20').seed_offset
    folds = {g: resolve_prior(g).fold for g in PUBLIC_GAMES}
    assert sorted(set(folds.values())) == [0, 1, 2, 3, 4]


def test_learner_never_receives_a_game_id():
    import inspect
    assert 'game' not in inspect.signature(DQNV1.__init__).parameters
