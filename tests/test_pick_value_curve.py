import pandas as pd

from src.features.pick_value_curve import fit_pick_value_curve


def test_curve_is_monotonically_non_increasing():
    df = pd.DataFrame(
        {
            "overall_pick": [1, 1, 2, 2, 5, 5, 10, 10, 50, 50, 100, 100],
            # Deliberately noisy - pick 5 has a higher raw average than pick 2 here,
            # which a real draft class could produce by chance (a late bloomer).
            # The fitted curve must still not increase with pick number.
            "point_shares": [50, 60, 20, 25, 40, 45, 10, 12, 3, 2, 1, 0],
        }
    )
    curve = fit_pick_value_curve(df)
    picks = [1, 2, 5, 10, 50, 100]
    preds = curve.predict(picks)
    assert all(preds[i] >= preds[i + 1] for i in range(len(preds) - 1))


def test_curve_clips_out_of_range_picks():
    df = pd.DataFrame({"overall_pick": [1, 10, 30], "point_shares": [50, 10, 2]})
    curve = fit_pick_value_curve(df)
    # pick 500 is beyond anything seen in training - should clip to the last known value, not extrapolate to negative.
    assert curve.predict([500])[0] >= 0
