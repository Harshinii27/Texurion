"""Performance -- real measured latency/memory benchmark."""

import os
import tempfile

import pandas as pd
import streamlit as st

from core.benchmark import (
    benchmark_all,
    build_benchmark_pdf,
    render_benchmark_chart,
)
from core.io import ensure_max_size, read_image

from .. import runtime, state

SAMPLE_PATH = os.path.join("images", "brick.jpeg")


def _fmt_mb(value):
    return "n/a" if value is None else f"{value:.1f} MB"


def render():
    st.markdown("## ⚡ Performance Benchmark")
    st.caption("Every figure is measured on this machine during this session "
               "(median over N runs; process peak working set). Numba speedups "
               "are only claimed where the equivalence check passes.")

    source = st.radio("Input source", ["Bundled sample", "Upload image"],
                      horizontal=True, index=0, key="bench_source")
    image = None
    if source == "Upload image":
        up = st.file_uploader("Upload image",
                              type=["jpg", "jpeg", "png", "bmp", "tif", "webp"],
                              key="bench_upload")
        if up is not None:
            image = runtime.decode_bytes(up.getvalue(), up.name)
    elif os.path.isfile(SAMPLE_PATH):
        image = read_image(SAMPLE_PATH)

    reps = st.slider("Repetitions per component", 1, 10, 3, key="bench_reps")

    if image is None:
        st.info("Choose an image to measure.")
        return

    work = ensure_max_size(image, 768)
    st.caption(f"Input {image.shape[1]}x{image.shape[0]}; benchmark runs at "
               f"{work.shape[1]}x{work.shape[0]}.")

    if st.button("Run benchmark", type="primary", key="bench_run"):
        with st.spinner("Measuring (this can take a while)…"):
            result = benchmark_all(image, reps=int(reps))

        st.subheader("Environment")
        st.dataframe(pd.DataFrame(
            [{"variable": k, "value": str(v)}
             for k, v in result.environment.items()]),
            width="stretch", hide_index=True)

        st.subheader(f"Latency (median over {result.reps} runs)")
        st.dataframe(pd.DataFrame(
            [{"component": name, "median_ms": round(s["median_ms"], 3),
              "mean_ms": round(s["mean_ms"], 3), "std_ms": round(s["std_ms"], 3)}
             for name, s in result.timings.items()]),
            width="stretch", hide_index=True)

        chart = render_benchmark_chart(
            result, os.path.join(tempfile.gettempdir(), "bench_chart.png"))
        st.image(chart, width="stretch")

        st.subheader("Memory")
        mem = result.memory
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline RSS", _fmt_mb(mem["baseline_rss_mb"]))
        m2.metric("Peak during run", _fmt_mb(mem["peak_rss_mb"]))
        m3.metric("Method", mem["method"])

        eq = result.equivalence
        if eq["equal"]:
            st.success("NumPy and Numba custom engines produced bit-identical "
                       "output; any speedup claim is backed by the measurements "
                       "above.")
        else:
            st.error(f"Engines differ ({eq['diff_count']} pixels); latencies are "
                     "informational only.")
        for note_text in result.notes:
            st.caption(f"* {note_text}")

        try:
            pdf = build_benchmark_pdf(
                result, chart,
                os.path.join(tempfile.gettempdir(), "benchmark_report.pdf"))
            with open(pdf, "rb") as handle:
                st.download_button("Download PDF report", data=handle.read(),
                                   file_name="benchmark_report.pdf",
                                   mime="application/pdf", key="bench_pdf")
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF export failed: {exc}")
