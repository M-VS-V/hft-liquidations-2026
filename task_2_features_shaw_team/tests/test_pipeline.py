from shaw_features import FeatureSet, default_features


def test_pipeline_generate_columns(ctx):
    fs = FeatureSet(default_features())
    df = fs.generate(ctx)
    assert "timestamp" in df.columns
    assert df.height == ctx.n_trades
    assert len(df.columns) == len(fs.names) + 1


def test_pipeline_validate_report_shape(ctx):
    fs = FeatureSet(default_features())
    rep = fs.validate(ctx)
    assert rep.height == len(fs.features)
    assert "ok" in rep.columns
