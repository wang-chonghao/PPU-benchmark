# Pinned reference source

`kernel.py`, `config.json` and `LICENSE` are unmodified bytes from
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/inference),
revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT license.
`provenance.json` records their SHA-256 hashes. This is DeepSeek's reference,
not an Alibaba-published PPU implementation or an official benchmark shape list.

The runner uses the original factory definitions, binds the declared symbolic
dimensions to each recorded test shape, and compiles explicitly for `ppu_15`
with the installed [SAIL TileLang](https://github.com/t-head/tilelang-for-sail).
The historical GTSim fixtures used compiler revision
`cd65f13dbffffef1d72fed6620c74886cb1ab877`. Do not silently change the original
algorithms, tiling, import dialect or FP4-to-FP8 implementation to make a failed
case pass. Record a separate variant and regenerate its paired IR if needed.
