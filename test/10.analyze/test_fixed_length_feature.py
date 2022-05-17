import freetensor as ft

STANDALONE_FEATURES = 0
SAMPLE_GROUPS = 10
SAMPLE_ITERS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
SAMPLE_FEATURES = 26
FEAT_SAMP_CPU_LOAD_AREA = 14


def test_sample_basic():
    with ft.VarDef([
        ("x", (128,), "float32", "input", "cpu"),
        ("y", (128, 32), "float32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 32) as j:
                with ft.For("k", 0, 32) as k:
                    y[i * 32 + k, j] = x[i * 32 + k] * 2
    ast = ft.pop_ast(verbose=True)

    features = ft.fixed_length_feature(ast)

    assert features[STANDALONE_FEATURES +
                    SAMPLE_ITERS.index(32) * SAMPLE_FEATURES +
                    FEAT_SAMP_CPU_LOAD_AREA] == 32  # Load 32 every 32 iters
    assert features[
        STANDALONE_FEATURES + SAMPLE_ITERS.index(1024) * SAMPLE_FEATURES +
        FEAT_SAMP_CPU_LOAD_AREA] == 32  # Load 32 every 32 * 32 iters
    assert features[STANDALONE_FEATURES +
                    SAMPLE_ITERS.index(4096) * SAMPLE_FEATURES +
                    FEAT_SAMP_CPU_LOAD_AREA] == 128  # Load 128 in total
