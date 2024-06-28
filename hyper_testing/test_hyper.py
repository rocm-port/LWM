
import torch
import torch2jax.torch2jax as torch2jax

import hyper_attn.models.attention.hyper_attn as hyperlib
import LWM.lwm.generic_random as generic_random
import LWM.lwm.hyper_attn as lwmh

seed=123815
torch.set_default_device("cuda")
rng = generic_random.generic_random(seed)
hyper = hyperlib.HyperAttention(rng=generic_random.generic_random(seed))
lwmh = lwmh.HyperAttention(rng=generic_random.generic_random(seed))

values = rng.normal((1, 12, 8192, 64), "float32")
tvalues = torch2jax.j2t(values)
hv = hyper.forward(tvalues, tvalues, tvalues).flatten()
jv = torch2jax.j2t(lwmh.forward(values, values, values)).flatten()

if not torch.allclose(hv, jv, atol=1e-5):
    diffs = jv-hv
    abspos = torch.argmax(diffs)
    reldiffs = diffs/hv
    relpos = torch.argmax(reldiffs)
    print(f"Mismatch! Absolute at {abspos}: {diffs[abspos]} ({hv[abspos]}:{jv[abspos]}) Relative at {relpos}: {reldiffs[relpos]} ({hv[relpos]}:{jv[relpos]})")
else:
    print("Match")
