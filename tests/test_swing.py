from fibtrader.swing import zigzag_confirmed


def test_zigzag_produces_alternating_pivots(synth_df):
    out = zigzag_confirmed(synth_df, pct=0.05, confirm_bars=2)
    pivots = out[out["pivot"] != 0]["pivot"].tolist()
    assert len(pivots) >= 2
    for a, b in zip(pivots[:-1], pivots[1:]):
        assert a + b == 0, "pivots must alternate sign"


def test_confirmation_after_pivot(synth_df):
    out = zigzag_confirmed(synth_df, pct=0.05, confirm_bars=3)
    piv_rows = out[out["pivot"] != 0]
    assert (piv_rows["pivot_confirmed_at"] >= piv_rows.index).all()


def test_no_pivots_with_huge_threshold(synth_df):
    out = zigzag_confirmed(synth_df, pct=10.0)
    assert (out["pivot"] == 0).all()
