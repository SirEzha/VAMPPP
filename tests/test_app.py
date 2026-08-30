"""Test application GUI structure and execution."""

import tkinter as tk
import pytest
from src.app import VoiceSimilarityApp


class TestApp:
    def test_gui_creation_and_widget_binding(self):
        root = tk.Tk()
        app = VoiceSimilarityApp(root)
        assert app.ref_path_var.get() == "No file loaded"
        assert app.headline_score_var.get() == "-- / 100"
        assert hasattr(app, "mic_combo")
        assert hasattr(app, "btn_record")
        assert hasattr(app, "details_tree")
        root.update_idletasks()
        root.update()
        del app
        root.destroy()
