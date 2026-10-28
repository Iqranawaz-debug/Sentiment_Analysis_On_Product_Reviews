"""Altair charts with fixed class colors and text labels. Altair ships with Streamlit."""

import altair as alt
import pandas as pd

from sentix.dataset import COLORS, LABELS

SCALE = alt.Scale(domain=list(LABELS), range=[COLORS[label] for label in LABELS])
# Review counts are whole numbers; without this, small counts get 0.2-step ticks.
COUNT_AXIS = alt.Axis(format=",d", tickMinStep=1)


def distribution(table: pd.DataFrame) -> alt.Chart:
    data = table.assign(text=[f"{c:,} ({s:.1%})" for c, s in zip(table["count"], table.share)])
    base = alt.Chart(data).encode(
        y=alt.Y("sentiment:N", sort=list(LABELS)[::-1], title=None),
        x=alt.X("count:Q", title="Reviews", axis=COUNT_AXIS),
        tooltip=[
            alt.Tooltip("sentiment:N", title="Sentiment"),
            alt.Tooltip("count:Q", title="Reviews", format=","),
            alt.Tooltip("share:Q", title="Share of analyzed", format=".1%"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("sentiment:N", scale=SCALE, legend=None)
    )
    labels = base.mark_text(align="left", dx=4, color="#172B3A").encode(text="text:N")
    return (bars + labels).properties(height=150)


def probabilities(values: dict) -> alt.Chart:
    data = pd.DataFrame({"sentiment": list(values), "probability": list(values.values())})
    base = alt.Chart(data).encode(
        y=alt.Y("sentiment:N", sort=list(LABELS)[::-1], title=None),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("sentiment:N"), alt.Tooltip("probability:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("sentiment:N", scale=SCALE, legend=None)
    )
    text = base.mark_text(align="left", dx=4, color="#172B3A").encode(
        text=alt.Text("probability:Q", format=".1%")
    )
    return (bars + text).properties(height=130)


def trend(table: pd.DataFrame, field: str) -> alt.Chart:
    title = "Share of analyzed reviews" if field == "share" else "Reviews"
    return (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y(
                f"{field}:Q",
                title=title,
                axis=alt.Axis(format="%") if field == "share" else COUNT_AXIS,
            ),
            color=alt.Color(
                "sentiment:N", scale=SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            strokeDash=alt.StrokeDash("sentiment:N", sort=list(LABELS), legend=None),
            tooltip=[
                alt.Tooltip("period:T", title="Period start"),
                alt.Tooltip("sentiment:N"),
                alt.Tooltip("count:Q", format=","),
                alt.Tooltip("share:Q", format=".1%"),
                alt.Tooltip("total:Q", title="Reviews in period", format=","),
            ],
        )
        .properties(height=260)
    )


def ratings(table: pd.DataFrame) -> alt.Chart:
    long = table.reset_index().melt(id_vars="rating", var_name="sentiment", value_name="count")
    long["total"] = long.groupby("rating")["count"].transform("sum")
    long["share"] = long["count"] / long.total.where(long.total > 0, 1)
    long["rating"] = long.rating.map(lambda r: f"{r:g}")
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y(
                "rating:N", title="Rating", sort="descending", axis=alt.Axis(labelOverlap=False)
            ),
            x=alt.X(
                "share:Q",
                stack="normalize",
                title="Predicted sentiment mix",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "sentiment:N", scale=SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            order=alt.Order("sentiment:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("rating:N"),
                alt.Tooltip("sentiment:N"),
                alt.Tooltip("count:Q", format=","),
                alt.Tooltip("share:Q", format=".1%", title="Share of this rating"),
            ],
        )
        .properties(height=alt.Step(34))
    )


def confidence(frame: pd.DataFrame) -> alt.Chart:
    data = frame[frame.status == "ok"][["sentiment", "confidence"]]
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "confidence:Q",
                bin=alt.Bin(extent=[0.3, 1.0], step=0.05),
                title="Predicted-class probability",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("count():Q", title="Reviews", axis=COUNT_AXIS),
            color=alt.Color(
                "sentiment:N", scale=SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            tooltip=[alt.Tooltip("sentiment:N"), alt.Tooltip("count():Q", title="Reviews")],
        )
        .properties(height=220)
    )


def confusion(matrix: list[list[int]], order: list[str]) -> alt.Chart:
    rows = [
        {"actual": actual, "predicted": predicted, "count": matrix[i][j]}
        for i, actual in enumerate(order)
        for j, predicted in enumerate(order)
    ]
    data = pd.DataFrame(rows)
    data["row_share"] = data["count"] / data.groupby("actual")["count"].transform("sum")
    base = alt.Chart(data).encode(
        x=alt.X("predicted:N", sort=order, title="Predicted"),
        y=alt.Y("actual:N", sort=order, title="Actual (rating-derived)"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("row_share:Q", scale=alt.Scale(scheme="teals"), title="Row share"),
        tooltip=[
            "actual",
            "predicted",
            alt.Tooltip("count:Q", format=","),
            alt.Tooltip("row_share:Q", format=".1%"),
        ],
    )
    text = base.mark_text().encode(
        text=alt.Text("count:Q", format=","),
        color=alt.condition("datum.row_share > 0.5", alt.value("white"), alt.value("#172B3A")),
    )
    return (cells + text).properties(height=240)
