from __future__ import annotations

import platform
import sys

import customtkinter as ctk
from tkinter import filedialog, messagebox, StringVar
from pipeline_calculator.gui.window import AppWindow

from pipeline_calculator.core.constants import (
    ANGULAR_TOLERANCE,
    DEFAULT_DETECTION_RANGE,
    MIN_PARALLEL_LENGTH,
    SEGMENT_LENGTH,
)
from pipeline_calculator.gui.actions.export_actions import export_with_dialog
from pipeline_calculator.gui.actions.open_kml_action import open_overlap_corridor
from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController
from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
from pipeline_calculator.gui.dialogs.params_dialog import ParamsDialog
from pipeline_calculator.gui.pages.file_select_page import show as show_file_select_page
from pipeline_calculator.gui.pages.results_page import show as show_results_page
from pipeline_calculator.gui.resources import set_window_icon
from pipeline_calculator.gui.state import AnalysisParameters, AppState
from pipeline_calculator.gui.preferences import StateBreakdownPreference
from pipeline_calculator.gui.repair_ui import RepairWorkflow


def _legacy_version() -> str:
    try:
        from pipeline_calculator import __version__

        return str(__version__)
    except Exception:
        return "4.0-dev.unknown"


class PipelineCalculatorGUI:
    """Refactored GUI entrypoint (Tk/CustomTkinter)."""

    def __init__(self) -> None:
        self.version = _legacy_version()

        self.root = AppWindow()
        set_window_icon(self.root)

        self.state = AppState()
        self.state_preference = StateBreakdownPreference(self.root)
        self.controller = AnalysisController()
        self._processing = False
        self._closing = False
        self._analysis_session = None
        self._params_dialog: ParamsDialog | None = None
        self.repair_workflow = RepairWorkflow(self.root, set_busy=self._set_processing,
            on_return=self.show_file_selection, on_replace=self.process_file, on_resume=self._start_analysis)

        # Keep StringVar for CTkEntry typing behavior (see legacy notes).
        self.detection_range_var = StringVar(value=str(DEFAULT_DETECTION_RANGE))
        self.min_parallel_var = StringVar(value=str(MIN_PARALLEL_LENGTH))
        self.segment_length_var = StringVar(value=str(SEGMENT_LENGTH))
        self.angular_tolerance_var = StringVar(value=str(ANGULAR_TOLERANCE))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.setup_gui()

    def setup_gui(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root.title(f"Pipeline Calculator v{self.version}")
        self.show_file_selection()
        self.root.initialize_size()
        self.root.deiconify()

    def show_file_selection(self) -> None:
        if self._processing:
            return
        show_file_select_page(
            self.root,
            title="Pipeline Calculator with Overlap Analysis",
            detection_range_var=self.detection_range_var,
            min_parallel_var=self.min_parallel_var,
            segment_length_var=self.segment_length_var,
            angular_tolerance_var=self.angular_tolerance_var,
            on_browse=self.browse_file,
            on_file_selected=self.process_file,
            retry_path=self.state.current_file,
            state_preference=self.state_preference,
            repair_workflow=getattr(self, 'repair_workflow', None),
            on_retry=lambda path: self.process_file(path, reuse_source=True),
        )

    def import_new_file(self):
        if self._processing:
            return
        self.repair_workflow.retire_source()
        self.state.current_file = None
        self.state.current_results = None
        self.show_file_selection()

    def _set_processing(self, busy):
        self._processing = busy
        preference = getattr(self, 'state_preference', None)
        if preference is not None:
            preference.set_busy(busy)

    def browse_file(self) -> None:
        if self._processing:
            return
        self.root.withdraw()
        try:
            filetypes = [
                ("All supported", "*.kmz *.kml"),
                ("KMZ files", "*.kmz"),
                ("KML files", "*.kml"),
                ("All files", "*.*"),
            ]
            file_path = filedialog.askopenfilename(filetypes=filetypes)
            if file_path:
                self.process_file(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to browse file: {str(e)}")
        finally:
            self.root.deiconify()

    def _get_parameters(self) -> AnalysisParameters:
        params, corrections = AnalysisParameters.from_strings(
            self.detection_range_var.get(),
            self.min_parallel_var.get(),
            self.segment_length_var.get(),
            self.angular_tolerance_var.get(),
        )
        # Keep input fields synchronized with corrected and clamped parameters.
        if "detection_range" in corrections:
            self.detection_range_var.set(corrections["detection_range"])
        if "min_parallel_length" in corrections:
            self.min_parallel_var.set(corrections["min_parallel_length"])
        if "segment_length" in corrections:
            self.segment_length_var.set(corrections["segment_length"])
        if "angular_tolerance" in corrections:
            self.angular_tolerance_var.set(corrections["angular_tolerance"])
        return params

    def process_file(self, file_path: str, *, reuse_source=False) -> None:
        if self._processing or getattr(self, '_closing', False):
            return
        self._set_processing(True)
        preference = getattr(self, 'state_preference', None)
        try:
            self.state.current_results = None
            self.state.current_file = file_path
            self.state.params = self._get_parameters()
            options = preference.snapshot() if preference is not None else None
            workflow = getattr(self, 'repair_workflow', None)
            if workflow is not None and not reuse_source:
                workflow.retire_source()
            source = workflow.session_for(file_path) if workflow is not None else None
            self._start_analysis(file_path, self.state.params, options=options, source_session=source)
        except Exception as e:
            self._processing = False
            if preference is not None:
                preference.set_busy(False)
            if getattr(self, '_analysis_session', None) is not None:
                self._analysis_session.close()
            messagebox.showerror("Processing Error", str(e))
            self.show_file_selection()

    def _start_analysis(self, path, params, *, options=None, source_session=None, approve_repair=False):
        self._set_processing(True)
        kwargs = {'options': options} if options is not None else {}
        if source_session is not None:
            kwargs['source_session'] = source_session
        if approve_repair:
            kwargs['approve_repair'] = True
        try:
            self._analysis_session = AnalysisSession(self.root, self._analysis_done, self.controller)
            self._analysis_session.start(path, params, **kwargs)
        except Exception as error:
            if self._analysis_session is not None:
                self._analysis_session.close()
            self._set_processing(False)
            messagebox.showerror('Processing Error', str(error), parent=self.root)
            self.show_file_selection()

    def _analysis_done(self, job):
        if getattr(self, '_closing', False):
            return
        if self._analysis_session is None or self._analysis_session.job is not job:
            return
        workflow = getattr(self, 'repair_workflow', None)
        if workflow is not None and workflow.handle_done(job):
            return
        self._set_processing(False)
        if job.state == 'completed':
            self.state.current_results = job.result
            self.show_results()
        else:
            if job.error is not None:
                prefix = 'File repair verified. Analysis could not finish.\n\n' if workflow is not None and workflow.verified else ''
                messagebox.showerror("Processing Error", prefix + str(job.error))
            self.show_file_selection()

    def close(self):
        self._closing = True
        if self._analysis_session is not None:
            self._analysis_session.close()
        if getattr(self, 'repair_workflow', None) is not None:
            self.repair_workflow.close()
        self.root.destroy()

    def show_results(self) -> None:
        if not self.state.current_results:
            self.show_file_selection()
            return

        show_results_page(
            self.root,
            version=self.version,
            current_file=self.state.current_file,
            current_results=self.state.current_results,
            on_export=self.export_results,
            on_reanalyze=self.reanalyze,
            on_new_file=self.import_new_file,
            on_exit=self.close,
            on_open_corridor=self.view_overlap_corridor,
            state_preference=self.state_preference,
            repair_workflow=getattr(self, 'repair_workflow', None),
        )

    def view_overlap_corridor(self, section: dict, index: int) -> None:
        from pipeline_calculator.gui.actions.corridor_launch import launch_corridor
        try:
            launch_corridor(self.root, section, index)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open KML file: {str(e)}")

    def reanalyze(self) -> None:
        if self._processing:
            return
        if self._params_dialog is not None:
            try:
                self._params_dialog.close()
            except Exception:
                pass
            self._params_dialog = None

        def apply_and_analyze():
            if not self.state.current_file:
                return
            self.process_file(self.state.current_file, reuse_source=True)

        self._params_dialog = ParamsDialog(
            self.root,
            detection_range_var=self.detection_range_var,
            segment_length_var=self.segment_length_var,
            min_parallel_var=self.min_parallel_var,
            angular_tolerance_var=self.angular_tolerance_var,
            on_apply=apply_and_analyze,
            state_preference=self.state_preference,
        )
        self._params_dialog.show()

    def export_results(self) -> None:
        if self._processing:
            return
        if not self.state.current_results:
            return
        export_with_dialog(self.state.current_results, self.state.current_file)

    def run(self) -> None:
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Application Error", f"Application error: {str(e)}")


def main() -> int:
    try:
        print(f"Pipeline Calculator v{_legacy_version()}")
        print(f"Running on {platform.system()} {platform.machine()}")
        print("-" * 50)

        app = PipelineCalculatorGUI()
        app.run()
        return 0
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        try:
            messagebox.showerror(
                "Fatal Error",
                "A fatal error occurred:\n\n"
                f"{str(e)}\n\n"
                "The application will now exit.",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
