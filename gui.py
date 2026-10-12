"""Texture Analysis Studio - a modern interactive GUI for LBP analysis.

Run with:  python gui.py
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

from analysis import analyze, previews
from classifier import TEXTURE_CLASSES
from core.io import SUPPORTED_EXTENSIONS
from core.lbp import LBP_METHODS

# --------------------------------------------------------------------------
# Visual identity
# --------------------------------------------------------------------------
ACCENT = "#4f8cff"
ACCENT_HOVER = "#3a7bf0"
SUCCESS = "#2ecc71"
WARNING = "#fbbf24"
DANGER = "#f87171"

CARD = ("#ffffff", "#171b24")
PANEL = ("#f2f5fb", "#10141d")
HEADER = ("#e7edfb", "#0d1119")
BORDER = ("#d9e1f0", "#2a3140")
TEXT = ("#1b2438", "#e8ecf5")
MUTED = ("#5b6b8c", "#98a3b3")

CLASS_COLORS = {
    "Smooth": "#38bdf8",
    "Fine": "#34d399",
    "Rough": "#fbbf24",
    "Coarse": "#f87171",
}


class ImageCard(ctk.CTkFrame):
    """A framed viewport that renders a PIL image scaled to fit."""

    def __init__(self, master, title, **kwargs):
        super().__init__(
            master,
            corner_radius=12,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            **kwargs,
        )
        self._title = title
        self._pil = None
        self._ctk_image = None
        self._render_job = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=MUTED,
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 2))

        self.image_label = ctk.CTkLabel(
            self, text="No image", text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.image_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 10))

        self.bind("<Configure>", self._schedule_render)

    def set_image(self, rgb_array):
        if rgb_array is None:
            self._pil = None
            self.image_label.configure(text="No image", image=None)
            return
        self._pil = Image.fromarray(np.asarray(rgb_array))
        self._render_now()

    def _schedule_render(self, _event=None):
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(50, self._render_now)

    def _render_now(self):
        self._render_job = None
        if self._pil is None:
            return

        width = max(80, self.winfo_width() - 24)
        height = max(80, self.winfo_height() - 46)

        image = self._pil.copy()
        image.thumbnail((width, height), Image.LANCZOS)

        self._ctk_image = ctk.CTkImage(
            light_image=image, dark_image=image, size=image.size
        )
        self.image_label.configure(image=self._ctk_image, text="")


class TextureAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Texture Analysis Studio - LBP")
        self.geometry("1480x880")
        self.minsize(1180, 720)

        self.current_path = None
        self.result = None
        self.previews = (None, None, None)
        self._auto_job = None
        self._queue = queue.Queue()

        self._build_layout()

        self.populate_samples()
        self._apply_default_sample()

        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header -------------------------------------------------------
        header = ctk.CTkFrame(
            self, fg_color=HEADER, corner_radius=0, height=76
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Texture Analysis Studio",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        )
        title.grid(row=0, column=0, sticky="w", padx=(24, 8), pady=8)

        subtitle = ctk.CTkLabel(
            header,
            text="Local Binary Pattern  |  GLCM  |  Statistical Texture Insight",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
        )
        subtitle.grid(row=0, column=0, sticky="w", padx=(24, 8), pady=(36, 8))

        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.grid(row=0, column=1, sticky="e", padx=16, pady=12)

        self.theme_switch = ctk.CTkSwitch(
            header_right,
            text="Dark mode",
            command=self._toggle_theme,
            progress_color=ACCENT,
            font=ctk.CTkFont(size=12),
        )
        self.theme_switch.pack(side="right", padx=4)
        self.theme_switch.select()

        # Main body ----------------------------------------------------
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.sidebar = self._build_sidebar(body)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(10, 4), pady=10)

        self.center = self._build_center(body)
        self.center.grid(row=0, column=1, sticky="nsew", padx=4, pady=10)

        self.results_panel = self._build_results(body)
        self.results_panel.grid(row=0, column=2, sticky="nsew", padx=(4, 10), pady=10)

        # Status bar ---------------------------------------------------
        status_bar = ctk.CTkFrame(self, fg_color=HEADER, corner_radius=0, height=36)
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.grid_columnconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(
            status_bar, text="Ready", text_color=MUTED,
            font=ctk.CTkFont(size=11), anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=16, pady=6)

        self.progress = ctk.CTkProgressBar(
            status_bar, width=220, progress_color=ACCENT
        )
        self.progress.grid(row=0, column=1, sticky="e", padx=16, pady=6)
        self.progress.set(0)

    def _build_sidebar(self, parent):
        frame = ctk.CTkScrollableFrame(
            parent,
            width=300,
            fg_color="transparent",
            scrollbar_button_color=ACCENT,
        )
        frame.grid_columnconfigure(0, weight=1)

        # --- Input card ---
        input_card = ctk.CTkFrame(frame, corner_radius=12, fg_color=CARD,
                                  border_width=1, border_color=BORDER)
        input_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        input_card.grid_columnconfigure(0, weight=1)

        self._section_title(input_card, "1. Load Image").grid(row=0, column=0, padx=14, pady=(12, 6))

        self.path_entry = ctk.CTkEntry(
            input_card, placeholder_text="No image selected",
            text_color=TEXT, fg_color=PANEL, border_color=BORDER,
        )
        self.path_entry.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        self.path_entry.configure(state="readonly")

        self.browse_btn = ctk.CTkButton(
            input_card, text="Browse Image", command=self._browse,
            corner_radius=8, height=34,
        )
        self.browse_btn.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))

        sample_row = ctk.CTkFrame(input_card, fg_color="transparent")
        sample_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        sample_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            sample_row, text="Sample:", text_color=MUTED, font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, padx=(0, 6))

        self.sample_menu = ctk.CTkOptionMenu(
            sample_row,
            values=["<none>"],
            command=self._on_sample_selected,
            fg_color=PANEL, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=TEXT, font=ctk.CTkFont(size=11),
            width=150, height=28,
        )
        self.sample_menu.grid(row=0, column=1, sticky="ew")

        # --- LBP parameters card ---
        params = ctk.CTkFrame(frame, corner_radius=12, fg_color=CARD,
                              border_width=1, border_color=BORDER)
        params.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        params.grid_columnconfigure(0, weight=1)

        self._section_title(params, "2. LBP Parameters").grid(row=0, column=0, padx=14, pady=(12, 6))

        self.radius_value = ctk.IntVar(value=3)
        self.points_value = ctk.IntVar(value=24)
        self.auto_points = ctk.BooleanVar(value=True)

        def _slider_row(label_text, var, low, high, step, callback):
            container = ctk.CTkFrame(params, fg_color="transparent")
            container.grid_columnconfigure(0, weight=1)
            name = ctk.CTkLabel(container, text=label_text, text_color=MUTED,
                                font=ctk.CTkFont(size=11))
            val = ctk.CTkLabel(container, text=str(var.get()), text_color=ACCENT,
                               font=ctk.CTkFont(size=12, weight="bold"))
            slider = ctk.CTkSlider(container, from_=low, to=high, number_of_steps=step,
                                   variable=var, command=callback,
                                   progress_color=ACCENT, button_color=ACCENT,
                                   button_hover_color=ACCENT_HOVER)
            name.grid(row=0, column=0, sticky="w")
            val.grid(row=0, column=1, sticky="e")
            slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
            return container, slider, val

        radius_row, radius_slider, radius_val = _slider_row(
            "Radius", self.radius_value, 1, 8, 7,
            lambda v: self._radius_changed(v))
        radius_row.grid(row=1, column=0, sticky="ew", padx=14, pady=4)
        self.radius_val_label = radius_val
        self.radius_slider = radius_slider

        def _points_row_build():
            container = ctk.CTkFrame(params, fg_color="transparent")
            container.grid_columnconfigure(0, weight=1)
            name = ctk.CTkLabel(container, text="Points", text_color=MUTED,
                                font=ctk.CTkFont(size=11))
            val = ctk.CTkLabel(container, text="auto", text_color=ACCENT,
                               font=ctk.CTkFont(size=12, weight="bold"))
            slider = ctk.CTkSlider(container, from_=8, to=64, number_of_steps=56,
                                   variable=self.points_value,
                                   command=lambda v: self._points_changed(v),
                                   progress_color=ACCENT, button_color=ACCENT,
                                   button_hover_color=ACCENT_HOVER)
            name.grid(row=0, column=0, sticky="w")
            val.grid(row=0, column=1, sticky="e")
            slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
            return container, slider, val

        points_container, points_slider, points_val = _points_row_build()
        points_container.grid(row=2, column=0, sticky="ew", padx=14, pady=4)
        self.points_val_label = points_val
        self.points_slider = points_slider
        self.points_slider.configure(state="disabled")

        auto_sw = ctk.CTkSwitch(
            params, text="Auto points (8 \u00d7 radius)",
            variable=self.auto_points, command=self._auto_points_toggled,
            font=ctk.CTkFont(size=11), progress_color=ACCENT,
        )
        auto_sw.grid(row=3, column=0, sticky="w", padx=14, pady=4)

        method_row = ctk.CTkFrame(params, fg_color="transparent")
        method_row.grid(row=4, column=0, sticky="ew", padx=14, pady=4)
        method_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(method_row, text="Method", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0, padx=(0, 6))
        self.method_menu = ctk.CTkOptionMenu(
            method_row, values=list(LBP_METHODS),
            command=lambda _m: self._schedule_auto_update(),
            fg_color=PANEL, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=TEXT, font=ctk.CTkFont(size=11), height=28,
        )
        self.method_menu.grid(row=0, column=1, sticky="ew")
        self.method_menu.set("uniform")

        analyze_btn = ctk.CTkButton(
            params, text="Analyze Texture", command=self._run_analysis,
            corner_radius=8, height=40, font=ctk.CTkFont(size=13, weight="bold"),
        )
        analyze_btn.grid(row=5, column=0, sticky="ew", padx=14, pady=(10, 14))
        self.analyze_btn = analyze_btn

        # --- Actions card ---
        actions = ctk.CTkFrame(frame, corner_radius=12, fg_color=CARD,
                               border_width=1, border_color=BORDER)
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        actions.grid_columnconfigure(0, weight=1)

        self._section_title(actions, "3. Export & Batch").grid(row=0, column=0, padx=14, pady=(12, 6))

        pdf_btn = ctk.CTkButton(
            actions, text="Export PDF Report", command=self._export_pdf,
            corner_radius=8, height=32,
            fg_color="transparent", border_width=1, border_color=ACCENT,
            text_color=ACCENT, hover_color=PANEL,
        )
        pdf_btn.grid(row=1, column=0, sticky="ew", padx=14, pady=4)

        save_btn = ctk.CTkButton(
            actions, text="Save Images & Data", command=self._save_outputs,
            corner_radius=8, height=32,
            fg_color="transparent", border_width=1, border_color=ACCENT,
            text_color=ACCENT, hover_color=PANEL,
        )
        save_btn.grid(row=2, column=0, sticky="ew", padx=14, pady=4)

        batch_btn = ctk.CTkButton(
            actions, text="Batch Process Folder", command=self._batch_process,
            corner_radius=8, height=32,
            fg_color="transparent", border_width=1, border_color=ACCENT,
            text_color=ACCENT, hover_color=PANEL,
        )
        batch_btn.grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 14))

        return frame

    def _section_title(self, parent, text):
        return ctk.CTkLabel(
            parent, text=text, text_color=TEXT,
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        )

    def _build_center(self, parent):
        center = ctk.CTkFrame(parent, fg_color="transparent")
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(0, weight=3)
        center.grid_rowconfigure(1, weight=2)

        # --- Preview strip ---
        previews_holder = ctk.CTkFrame(center, fg_color="transparent")
        previews_holder.grid(row=0, column=0, sticky="nsew")
        for i in range(3):
            previews_holder.grid_columnconfigure(i, weight=1)

        self.card_original = ImageCard(previews_holder, "Original Image")
        self.card_original.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self.card_gray = ImageCard(previews_holder, "Grayscale")
        self.card_gray.grid(row=0, column=1, sticky="nsew", padx=4)

        self.card_lbp = ImageCard(previews_holder, "LBP Feature Map")
        self.card_lbp.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        # --- Histogram card ---
        hist_card = ctk.CTkFrame(center, corner_radius=12, fg_color=CARD,
                                 border_width=1, border_color=BORDER)
        hist_card.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        hist_card.grid_columnconfigure(0, weight=1)
        hist_card.grid_rowconfigure(1, weight=1)

        hist_header = ctk.CTkFrame(hist_card, fg_color="transparent")
        hist_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 2))
        ctk.CTkLabel(
            hist_header, text="LBP Histogram", text_color=TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left")
        self.hist_info_label = ctk.CTkLabel(
            hist_header, text="", text_color=MUTED, font=ctk.CTkFont(size=11),
        )
        self.hist_info_label.pack(side="right")

        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(6, 2.6), dpi=100, tight_layout=True)
        self.hist_ax = self.figure.add_subplot(111)
        self._style_axes(self.hist_ax)

        canvas_host = ctk.CTkFrame(hist_card, fg_color=PANEL, corner_radius=8)
        canvas_host.grid(row=1, column=0, sticky="nsew", padx=14, pady=(2, 12))
        canvas_host.grid_columnconfigure(0, weight=1)
        canvas_host.grid_rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_host)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._draw_empty_hist()
        return center

    def _style_axes(self, ax):
        dark = ctk.get_appearance_mode() == "Dark"
        bg = "#10141d" if dark else "#f2f5fb"
        gridc = "#2a3140" if dark else "#d9e1f0"
        textc = "#98a3b3" if dark else "#5b6b8c"
        ax.set_facecolor(bg)
        ax.tick_params(colors=textc, labelsize=8)
        ax.grid(axis="y", color=gridc, linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(gridc)
        ax.set_xlabel("LBP pattern bin", color=textc, fontsize=9)
        ax.set_ylabel("Frequency", color=textc, fontsize=9)

    def _build_results(self, parent):
        panel = ctk.CTkScrollableFrame(
            parent, width=340, fg_color="transparent",
            scrollbar_button_color=ACCENT,
        )
        panel.grid_columnconfigure(0, weight=1)

        # --- Classification card ---
        class_card = ctk.CTkFrame(panel, corner_radius=12, fg_color=CARD,
                                  border_width=1, border_color=BORDER)
        class_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        class_card.grid_columnconfigure(0, weight=1)

        self._section_title(class_card, "Classification Result").grid(
            row=0, column=0, padx=14, pady=(12, 4))

        self.class_label = ctk.CTkLabel(
            class_card, text="--", font=ctk.CTkFont(size=26, weight="bold"),
            text_color=MUTED,
        )
        self.class_label.grid(row=1, column=0, sticky="ew", padx=14)

        self.conf_badge = ctk.CTkLabel(
            class_card, text="Awaiting analysis", text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.conf_badge.grid(row=2, column=0, sticky="ew", padx=14, pady=(2, 4))

        self.conf_bar = ctk.CTkProgressBar(class_card, height=10, progress_color=ACCENT)
        self.conf_bar.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.conf_bar.set(0)

        self.class_score_rows = {}
        class_scores_host = ctk.CTkFrame(class_card, fg_color="transparent")
        class_scores_host.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        class_scores_host.grid_columnconfigure(0, weight=1)
        for idx, name in enumerate(TEXTURE_CLASSES):
            row = ctk.CTkFrame(class_scores_host, fg_color="transparent")
            row.grid(row=idx, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=name, text_color=MUTED,
                         font=ctk.CTkFont(size=10), width=46, anchor="w"
                         ).grid(row=0, column=0, sticky="w")
            bar = ctk.CTkProgressBar(row, height=6, progress_color=CLASS_COLORS[name])
            bar.grid(row=0, column=1, sticky="ew")
            pct = ctk.CTkLabel(row, text="0%", text_color=MUTED,
                               font=ctk.CTkFont(size=9), width=38, anchor="e")
            pct.grid(row=0, column=2, padx=(6, 0))
            self.class_score_rows[name] = (bar, pct)

        self.interpret_label = ctk.CTkLabel(
            class_card, text="Upload an image to start.",
            text_color=MUTED, font=ctk.CTkFont(size=11), wraplength=300, justify="left",
        )
        self.interpret_label.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))

        # --- Metrics card ---
        metrics_card = ctk.CTkFrame(panel, corner_radius=12, fg_color=CARD,
                                    border_width=1, border_color=BORDER)
        metrics_card.grid(row=1, column=0, sticky="ew")
        metrics_card.grid_columnconfigure(0, weight=1)
        metrics_card.grid_columnconfigure(1, weight=1)

        self._section_title(metrics_card, "Texture Metrics").grid(
            row=0, column=0, columnspan=2, padx=14, pady=(12, 6))

        self.metric_labels = {}
        metric_names = [
            ("Contrast", "Contrast"),
            ("Dissimilarity", "Dissimilarity"),
            ("Homogeneity", "Homogeneity"),
            ("Energy", "Energy"),
            ("ASM", "ASM"),
            ("Correlation", "Correlation"),
            ("Entropy", "Entropy"),
            ("Uniformity", "Uniformity"),
            ("Mean", "Mean"),
            ("Std Dev", "Std Dev"),
        ]
        for i, (key, display) in enumerate(metric_names):
            row = i // 2
            col = i % 2
            cell = ctk.CTkFrame(metrics_card, fg_color=PANEL, corner_radius=8)
            cell.grid(row=row + 1, column=col, sticky="ew", padx=8, pady=4)
            cell.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(cell, text=display, text_color=MUTED,
                         font=ctk.CTkFont(size=10), anchor="w"
                         ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
            value = ctk.CTkLabel(cell, text="--", text_color=TEXT,
                                 font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
            value.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
            self.metric_labels[key] = value

        return panel

    # ------------------------------------------------------------------
    # Samples / file loading
    # ------------------------------------------------------------------
    def populate_samples(self):
        samples = []
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if os.path.isdir(images_dir):
            for name in sorted(os.listdir(images_dir)):
                if name.lower().endswith(SUPPORTED_EXTENSIONS):
                    samples.append(os.path.join(images_dir, name))
        self.sample_paths = samples
        names = [os.path.basename(p) for p in samples] or ["<none>"]
        self.sample_menu.configure(values=names)
        if samples:
            self.sample_menu.set(names[0])
        else:
            self.sample_menu.set("<none>")

    def _apply_default_sample(self):
        if self.sample_paths:
            self._on_sample_selected(os.path.basename(self.sample_paths[0]))

    def _on_sample_selected(self, name):
        if name in ("<none>",):
            return
        path = next((p for p in self.sample_paths if os.path.basename(p) == name), None)
        if path:
            self._set_current_path(path)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if path:
            self._set_current_path(path)

    def _set_current_path(self, path):
        self.current_path = path
        self.path_entry.configure(state="normal")
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, path)
        self.path_entry.configure(state="readonly")
        self.status_label.configure(text=f"Loaded: {os.path.basename(path)}")
        self._run_analysis()

    # ------------------------------------------------------------------
    # Parameter events
    # ------------------------------------------------------------------
    def _radius_changed(self, value):
        self.radius_value.set(int(float(value)))
        self.radius_val_label.configure(text=str(int(float(value))))
        if self.auto_points.get():
            self.points_value.set(8 * int(float(value)))
            self.points_val_label.configure(text=str(8 * int(float(value))))
        self._schedule_auto_update()

    def _points_changed(self, value):
        self.points_value.set(int(float(value)))
        self.points_val_label.configure(text=str(int(float(value))))
        self._schedule_auto_update()

    def _auto_points_toggled(self):
        if self.auto_points.get():
            self.points_value.set(8 * self.radius_value.get())
        self.points_val_label.configure(
            text="auto" if self.auto_points.get() else str(self.points_value.get()))
        self.points_slider.configure(state="normal" if not self.auto_points.get() else "disabled")
        self._schedule_auto_update()

    def _schedule_auto_update(self):
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
        self._auto_job = self.after(250, self._run_analysis)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _current_params(self):
        radius = self.radius_value.get()
        points = None if self.auto_points.get() else self.points_value.get()
        return radius, points, self.method_menu.get()

    def _run_analysis(self):
        self._auto_job = None
        if not self.current_path:
            self._set_status("Load an image to analyze.", WARNING)
            return

        radius, points, method = self._current_params()
        if self.analyze_btn:
            self.analyze_btn.configure(state="disabled")
        self.progress.set(0.05)
        self._set_status("Analyzing...", ACCENT)

        def worker():
            try:
                result = analyze(self.current_path, radius=radius, points=points, method=method)
                previews_images = previews(result)
                self._queue.put(("result", result, previews_images))
            except Exception as exc:  # noqa: BLE001
                self._queue.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "result":
                    self._apply_result(item[1], item[2])
                elif kind == "error":
                    self._on_error(item[1])
                elif kind == "batch_progress":
                    self.progress.set(item[1])
                    self._set_status(
                        f"Batched {item[2]}/{item[3]}: {item[4]}", ACCENT)
                elif kind == "batch_done":
                    self._finish_batch(*item[1:])
        except queue.Empty:
            pass

        if self._exists:
            self.after(50, self._poll_queue)

    def _apply_result(self, result, previews_images):
        self.result = result
        self.previews = previews_images

        original, gray, lbp = previews_images
        self.card_original.set_image(original)
        self.card_gray.set_image(gray)
        self.card_lbp.set_image(lbp)

        self._update_histogram()
        self._update_result_panel()

        self.analyze_btn.configure(state="normal")
        self.progress.set(1)
        self._set_status(
            f"{result.name} -> {result.label} ({result.confidence:.1f}%)", SUCCESS)

    def _on_error(self, exc):
        self.analyze_btn.configure(state="normal")
        self.progress.set(0)
        self._set_status(f"Error: {exc}", DANGER)
        messagebox.showerror("Analysis failed", str(exc))

    def _update_histogram(self):
        if self.result is None:
            return
        result = self.result
        self.hist_ax.clear()
        self._style_axes(self.hist_ax)

        dark = ctk.get_appearance_mode() == "Dark"
        bar_color = ACCENT if dark else "#2d6fe8"
        line_color = "#ffd166"

        x = np.arange(result.hist.size)
        self.hist_ax.bar(x, result.hist, width=0.85, color=bar_color, edgecolor="none")
        self.hist_ax.plot(x, result.hist, color=line_color, linewidth=1.2, alpha=0.9)
        self.hist_ax.set_title(
            f"{result.method} | radius={result.radius}, points={result.points}",
            color="#98a3b3" if dark else "#5b6b8c", fontsize=9,
        )

        entropy = result.features.get("Entropy", 0.0)
        uniformity = result.features.get("Uniformity", 0.0)
        self.hist_info_label.configure(
            text=f"{result.nbins} bins   |   entropy {entropy:.3f}   |   uniformity {uniformity:.3f}")

        self.figure.canvas.draw_idle()

    def _draw_empty_hist(self):
        self.hist_ax.clear()
        self._style_axes(self.hist_ax)
        self.hist_ax.text(
            0.5, 0.5, "Histogram will appear here",
            ha="center", va="center", transform=self.hist_ax.transAxes,
            color=("#98a3b3" if ctk.get_appearance_mode() == "Dark" else "#5b6b8c"),
            fontsize=11,
        )
        self.hist_ax.set_xlabel("")
        self.hist_ax.set_ylabel("")
        self.figure.canvas.draw_idle()

    def _update_result_panel(self):
        result = self.result
        if result is None:
            return

        color = CLASS_COLORS.get(result.label, ACCENT)
        self.class_label.configure(text=result.label, text_color=color)
        self.conf_badge.configure(
            text=f"Confidence {result.confidence:.1f}%   |   roughness {result.classification.get('roughness', 0):.3f}",
            text_color=color,
        )
        self.conf_bar.configure(progress_color=color)
        self.conf_bar.set(max(0.02, result.confidence / 100.0))

        for name, (bar, pct) in self.class_score_rows.items():
            prob = result.classification.get("scores", {}).get(name, 0.0)
            bar.set(max(0.02, prob))
            pct.configure(text=f"{prob * 100:.0f}%")

        self.interpret_label.configure(
            text=result.interpretation,
            text_color=TEXT if result.confidence >= 60 else MUTED,
        )

        for key, label in self.metric_labels.items():
            value = result.features.get(key)
            if value is None:
                continue
            if abs(value) >= 1000:
                text = f"{value:.2e}"
            elif abs(value) >= 10:
                text = f"{value:.2f}"
            else:
                text = f"{value:.4f}"
            label.configure(text=text)

    def _set_status(self, message, color=MUTED):
        self.status_label.configure(text=message, text_color=color)

    def _toggle_theme(self):
        if ctk.get_appearance_mode().lower() == "dark":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")

        target_selected = ctk.get_appearance_mode().lower() == "dark"
        if (self.theme_switch.get() == 1) != target_selected:
            if target_selected:
                self.theme_switch.select()
            else:
                self.theme_switch.deselect()

        for ax in (self.hist_ax,):
            self._style_axes(ax)
        self._redraw_figure()

    def _redraw_figure(self):
        if self.result is None:
            self._draw_empty_hist()
        else:
            self._update_histogram()

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _require_result(self):
        if self.result is None:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return False
        return True

    def _base_dirs(self):
        return (
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"),
        )

    def _export_pdf(self):
        if not self._require_result():
            return
        import report
        outputs_dir, reports_dir = self._base_dirs()
        original, gray, lbp = self.previews
        try:
            self._set_status("Building report...", ACCENT)
            paths = report.save_bundle(self.result, outputs_dir, original, gray, lbp)
            pdf = report.build_pdf_report(self.result, paths, reports_dir, paths["histogram"])
            self._set_status(f"Exported PDF: {pdf}", SUCCESS)
            os.startfile(pdf)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Export failed: {exc}", DANGER)
            messagebox.showerror("Export failed", str(exc))

    def _save_outputs(self):
        if not self._require_result():
            return
        import report
        outputs_dir, _ = self._base_dirs()
        original, gray, lbp = self.previews
        try:
            self._set_status("Saving images and data...", ACCENT)
            paths = report.save_bundle(self.result, outputs_dir, original, gray, lbp)
            self._set_status(f"Saved {len(paths)} artefacts to outputs/", SUCCESS)
            messagebox.showinfo("Saved",
                                "Saved:\n" + "\n".join(f"- {p}" for p in paths.values()))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Save failed: {exc}", DANGER)
            messagebox.showerror("Save failed", str(exc))

    def _batch_process(self):
        folder = filedialog.askdirectory(title="Select a folder of images")
        if not folder:
            return

        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )
        if not files:
            messagebox.showinfo("No images", "No supported images found in that folder.")
            return

        import report
        from datetime import datetime
        outputs_dir, reports_dir = self._base_dirs()
        batch_dir = os.path.join(
            outputs_dir, "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        radius, points, method = self._current_params()

        self.progress.set(0)
        total = len(files)
        self._set_status(f"Processing {total} images...", ACCENT)

        def worker():
            results = []
            errors = []
            total_batch = total
            for i, filename in enumerate(files, start=1):
                path = os.path.join(folder, filename)
                try:
                    result = analyze(path, radius=radius, points=points, method=method)
                    results.append(result)
                    previews_images = previews(result)
                    original, gray, lbp = previews_images
                    report.save_bundle(result, batch_dir, original, gray, lbp)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{filename}: {exc}")

                self._queue.put(("batch_progress", i / total_batch, i,
                                 total_batch, filename))

            csv_path = report.batch_csv(results, reports_dir)
            self._queue.put(("batch_done", total_batch, len(results),
                             errors, csv_path, batch_dir))

        threading.Thread(target=worker, daemon=True).start()

    @property
    def _exists(self):
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    def _finish_batch(self, total, ok, errors, csv_path, batch_dir):
        self.progress.set(1)
        if errors:
            self._set_status(f"Batch done: {ok}/{total} succeeded, {len(errors)} failed", WARNING)
            messagebox.showwarning(
                "Batch finished",
                f"Processed {ok}/{total}.\n\nReport:\n{csv_path}\n\nErrors:\n"
                + "\n".join(errors[:8]),
            )
        else:
            self._set_status(f"Batch done: {ok} images exported", SUCCESS)
            messagebox.showinfo(
                "Batch finished",
                f"Processed {ok} images.\n\nOutput: {batch_dir}\n\nCSV: {csv_path}",
            )


def main():
    app = TextureAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()