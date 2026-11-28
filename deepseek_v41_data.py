"""Deterministic CPU inputs and mathematical checks; never inside timed regions."""


def fp4_round(x):
    import torch
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    distance = (x.abs().unsqueeze(-1) - lut).abs()
    # IEEE nearest-even: prefer an even encoding if two magnitudes tie.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    codes = order[distance[..., order].argmin(-1)]
    codes = codes | ((x < 0).long() << 3)
    values = lut[codes & 7] * torch.where((codes & 8) != 0, -1., 1.)
    return values, codes.to(torch.uint8)


def unpack_fp4(packed):
    import torch
    raw = packed.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    return lut[codes & 7] * torch.where((codes & 8) != 0, -1., 1.)


def pack_fp4(codes):
    import torch
    # Same raw-byte convention as SAIL's PPU FP4 example: even K is low nibble.
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.int8)


def scale_tensor(values, dtype):
    import torch
    if dtype == "float8_e8m0fnu":
        raw = (torch.log2(values).round().long() + 127).to(torch.uint8)
        return raw.view(torch.float8_e8m0fnu)
    return values.to(torch.float8_e4m3fn)


def decode_scale(tensor, dtype):
    import torch
    if dtype == "float8_e8m0fnu":
        raw = tensor.view(torch.uint8)
        value = torch.pow(2., raw.float() - 127)
        return torch.where(raw == 255, float("nan"), value)
    return tensor.float()


def make_data(case, seed):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    s, o, family = case["shapes"], case["options"], case["family"]
    # Match the upstream wrapper / TileLang ABI when Torch has native x2 storage.
    fp4_storage = getattr(torch, "float4_e2m1fn_x2", torch.int8)

    def rand(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32)

    def output(shape, dtype):
        x = torch.empty(shape, dtype=dtype)
        x.view(torch.uint8).fill_(255)
        return x

    if family in ("act_quant_kernel", "fp4_quant_kernel"):
        m, n, group = s["M"], s["N"], o["group"]
        x = rand((m, n)).to(torch.bfloat16)
        x[0, :group] = 0  # Explicit zero group checks lower-bound scale handling.
        blocks = x.float().reshape(m, n // group, group)
        amax = blocks.abs().amax(-1)
        is_fp4 = family == "fp4_quant_kernel"
        maximum = 6 if is_fp4 else 448
        if is_fp4 and o["scale"] == "float8_e4m3":
            scales = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
        else:
            lower = 6 * 2**-126 if is_fp4 else 1e-4
            scales = torch.pow(2., torch.ceil(torch.log2(amax.clamp_min(lower) / maximum)))
        normalized = (blocks / scales.unsqueeze(-1)).clamp(-maximum, maximum)
        if is_fp4:
            quantized, codes = fp4_round(normalized)
        else:
            quantized = normalized.to(torch.float8_e4m3fn).float()
        if o["inplace"]:
            y_ref = (quantized * scales.unsqueeze(-1)).reshape(m, n).to(torch.bfloat16).float()
            y = output((m, n), torch.bfloat16)
        elif is_fp4:
            y_ref = quantized.reshape(m, n)
            y = output((m, n // 2), fp4_storage)
        else:
            y_ref = quantized.reshape(m, n)
            y = output((m, n), torch.float8_e4m3fn)
        scale_dtype = torch.float8_e8m0fnu if o["scale"] == "float8_e8m0fnu" else torch.float8_e4m3fn
        return [x, y, output(scales.shape, scale_dtype)], [1, 2], [y_ref, scales]

    if family in ("fp8_gemm_kernel", "fp4_gemm_kernel"):
        m, n, k, group = s["M"], s["N"], s["K"], o["group"]
        a = rand((m, k)).to(torch.float8_e4m3fn)
        if family == "fp4_gemm_kernel":
            b_ref, codes = fp4_round(rand((n, k)).clamp(-6, 6))
            b = pack_fp4(codes).view(fp4_storage)
            sb = torch.pow(2., torch.randint(-2, 2, (n, k // 32), generator=g).float())
            b_dequant = b_ref * sb.repeat_interleave(32, -1)
        else:
            b = rand((n, k)).to(torch.float8_e4m3fn)
            sb = torch.pow(2., torch.randint(-2, 2, (n // group, k // group), generator=g).float())
            b_dequant = b.float() * sb.repeat_interleave(group, 0).repeat_interleave(group, 1)
        sa = torch.pow(2., torch.randint(-2, 2, (m, k // group), generator=g).float())
        reference = (a.float() * sa.repeat_interleave(group, -1)) @ b_dequant.T
        return [a, b, output((m, n), torch.bfloat16), scale_tensor(sa, "float8_e8m0fnu"),
                scale_tensor(sb, "float8_e8m0fnu")], [2], [reference.to(torch.bfloat16).float()]

    if family == "hc_split_sinkhorn_kernel":
        n, hc, eps = s["n"], s["hc"], o["eps"]
        width = (2 + hc) * hc
        mixes, scale, base = rand((n, width)), rand((3,)), rand((width,))
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[:, hc:2*hc] * scale[1] + base[hc:2*hc])
        comb = (mixes[:, 2*hc:] * scale[2] + base[2*hc:]).reshape(n, hc, hc).softmax(-1) + eps
        comb /= comb.sum(-2, keepdim=True) + eps
        for _ in range(o["iters"] - 1):
            comb /= comb.sum(-1, keepdim=True) + eps
            comb /= comb.sum(-2, keepdim=True) + eps
        return [mixes, scale, base, output(pre.shape, torch.float32), output(post.shape, torch.float32),
                output(comb.shape, torch.float32)], [3, 4, 5], [pre, post, comb]

    if family == "sparse_attn_kernel":
        b, m, n, h, d, topk = (s[k] for k in ("b", "m", "n", "h", "d", "topk"))
        q, kv = rand((b, m, h, d)).to(torch.bfloat16), rand((b, n, d)).to(torch.bfloat16)
        sink = rand((h,))
        if o["pattern"] == "repeat_pad":
            row = torch.tensor([-1 if i % 7 == 0 else 512 + (i * 11) % 57 for i in range(topk)], dtype=torch.int32)
            indices = row.expand(b, m, topk).clone()
        elif o["pattern"] == "all_invalid":
            indices = torch.full((b, m, topk), -1, dtype=torch.int32)
        else:
            indices = torch.randint(0, n, (b, m, topk), generator=g, dtype=torch.int32)
            indices[..., ::17] = -1
        reference = torch.empty_like(q, dtype=torch.float32)
        # Independent dense reference with explicit sink probability and invalid masking.
        for bi in range(b):
            for mi in range(m):
                ix = indices[bi, mi].long()
                keys = kv[bi, ix.clamp_min(0)].float()
                keys[ix < 0] = 0
                scores = q[bi, mi].float() @ keys.T / d**.5
                scores[:, ix < 0] = -float("inf")
                prob = torch.cat((scores, sink[:, None]), -1).softmax(-1)[:, :-1]
                reference[bi, mi] = prob @ keys
        return [q, kv, output(q.shape, torch.bfloat16), sink, indices], [2], [reference.to(torch.bfloat16).float()]
    raise ValueError("unknown family: " + family)


def check_outputs(case, actual, expected):
    import torch
    family, options = case["family"], case["options"]
    if family in ("act_quant_kernel", "fp4_quant_kernel"):
        y = unpack_fp4(actual[0]) if family == "fp4_quant_kernel" and not options["inplace"] else actual[0].float()
        actual = [y, decode_scale(actual[1], options["scale"])]
        # Power-of-two scaling and explicitly rounded formats should agree exactly.
        # A backend rounding difference must be reviewed, not silently accepted.
        tolerances = [(0., 0.), (0., 0.)]
    elif family == "sparse_attn_kernel":
        tolerances = [(0.02, 0.002)]  # upstream casts online-softmax tiles to BF16
    elif family == "hc_split_sinkhorn_kernel":
        tolerances = [(2e-4, 2e-5)] * 3
    else:
        tolerances = [(0.01, 0.05)]
    if len(actual) != len(expected) or len(actual) != len(tolerances):
        raise ValueError("output/reference count mismatch")
    checks = []
    for got, want, (rtol, atol) in zip(actual, expected, tolerances):
        got, want = got.float(), want.float()
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol, equal_nan=False)
        checks.append(dict(max_abs_error=float((got-want).abs().max()), rtol=rtol, atol=atol))
    return dict(status="passed", reference="independent CPU mathematical reference", outputs=checks)
