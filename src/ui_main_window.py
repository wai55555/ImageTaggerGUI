from typing import TYPE_CHECKING

from PySide6.QtCore import (
    Qt
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSizePolicy, QSplitter, QStackedWidget, QComboBox, QButtonGroup,
    QCheckBox
)
from PySide6.QtGui import (
    QIcon
)


import constants
import model_registry

from custom_widgets import PathLineEdit, TagListWidget
from custom_dialogs import ClickableLabel
from grid_view_widget import GridViewWidget

if TYPE_CHECKING:
    from main_window import MainWindow

class Ui_MainWindow(object):
    def setup_ui(self, main_window: 'MainWindow'):
        main_window.setWindowTitle(constants.MSG_WINDOW_TITLE)
        main_window.setWindowIcon(QIcon(str(constants.RESOURCE_DIR / "icons/app_icon.ico")))
        try:
            if main_window.settings.window and main_window.settings.window.geometry:
                geometry_str = main_window.settings.window.geometry
                geom = geometry_str.split('+')
                size = geom[0].split('x')
                main_window.resize(int(size[0]), int(size[1]))
                main_window.move(int(geom[1]), int(geom[2]))
            else:
                raise ValueError("Geometry is not set")
        except (ValueError, IndexError):
            main_window.resize(950, 720)
            main_window.move(50, 50)
        
        main_window.setAcceptDrops(True)

        # Central stacked widget for view switching
        main_window.central_widget = QStackedWidget(main_window)
        main_window.setCentralWidget(main_window.central_widget)

        # Main View Setup
        main_window.main_view_widget = self._create_main_view(main_window)
        main_window.central_widget.addWidget(main_window.main_view_widget)

        # Grid View Setup
        main_window.grid_view_widget = GridViewWidget(main_window.settings, main_window.locale_manager)
        main_window.central_widget.addWidget(main_window.grid_view_widget)
        
        self._connect_signals(main_window)
        main_window._check_model_status_and_update_ui(auto_start_download=True)  # type: ignore

    def _connect_signals(self, main_window: 'MainWindow'):
        """Connects all signals to their corresponding slots."""
        main_window.image_list.itemClicked.connect(main_window.select_image_item)
        main_window.input_line.editingFinished.connect(main_window._on_input_path_changed)
        main_window.input_line.textChanged.connect(main_window._update_input_dir)  # type: ignore
        main_window.input_line.folder_dropped.connect(main_window._handle_folder_drop)  # type: ignore
        main_window.run_button.clicked.connect(main_window.toggle_download_or_start_tagging)
        main_window.grid_view_widget.back_to_main_requested.connect(main_window._show_main_view)  # type: ignore
        main_window.grid_view_widget.undo_button.clicked.connect(main_window._perform_undo)  # type: ignore
        main_window.grid_view_widget.redo_button.clicked.connect(main_window._perform_redo)  # type: ignore
        main_window.grid_view_widget.tags_added.connect(main_window._on_gridview_tags_added)  # type: ignore
        main_window.grid_view_widget.tag_removed.connect(main_window._on_gridview_tag_removed)  # type: ignore
        main_window.grid_view_widget.caption_edited.connect(main_window._on_gridview_caption_edited)  # type: ignore
        main_window.grid_view_widget.caption_save_failed.connect(main_window._on_gridview_caption_save_failed)  # type: ignore
        main_window.grid_view_widget.tag_hovered.connect(main_window._highlight_files_for_tag)  # type: ignore
        main_window.grid_view_widget.tag_hover_cleared.connect(main_window._clear_highlight)  # type: ignore
        main_window._resize_timer.timeout.connect(main_window._handle_resize_debounced)  # type: ignore

    def _create_main_view(self, main_window: 'MainWindow') -> QWidget:
        """Constructs the main view widget with its layout and components."""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        main_window.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = self._create_left_panel(main_window)
        right_panel = self._create_right_panel(main_window)

        main_window.splitter.addWidget(left_panel)
        main_window.splitter.addWidget(right_panel)
        main_window.splitter.setStretchFactor(0, 1)
        main_window.splitter.setStretchFactor(1, 3)
        
        layout.addWidget(main_window.splitter)
        return main_widget

    def _create_left_panel(self, main_window: 'MainWindow') -> QWidget:
        """Creates the left panel containing the image list."""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        main_window.image_list = TagListWidget(get_string=main_window.locale_manager.get_string)
        main_window.image_list.setMaximumWidth(500)
        layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Image_File_List")))
        layout.addWidget(main_window.image_list)
        return left_widget

    def _create_right_panel(self, main_window: 'MainWindow') -> QWidget:
        """Creates the right panel with all the controls and viewers."""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)

        main_window.right_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        input_group = self._create_input_group(main_window)
        viewer_group = self._create_viewer_group(main_window)
        bulk_actions_group = self._create_bulk_actions_group(main_window)
        log_group = self._create_log_group(main_window)

        layout.addWidget(input_group)
        main_window.right_vertical_splitter.addWidget(viewer_group)
        main_window.right_vertical_splitter.addWidget(bulk_actions_group)
        main_window.right_vertical_splitter.addWidget(log_group)
        
        # 初期は tagger レイアウト。captioner / VLM 切替時は
        # MainWindow._rebalance_right_splitter が constants の値へ組み替える。
        main_window.right_vertical_splitter.setSizes(constants.RIGHT_SPLIT_TAGGER)
        main_window.right_vertical_splitter.setStretchFactor(0, 4)
        main_window.right_vertical_splitter.setStretchFactor(1, 2)
        main_window.right_vertical_splitter.setStretchFactor(2, 1)

        layout.addWidget(main_window.right_vertical_splitter)
        return right_widget

    def _create_input_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the input folder selection group box."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Input_Folder"), main_window)
        layout = QVBoxLayout(group)
        controls_layout = QHBoxLayout()
        
        main_window.input_line = PathLineEdit(main_window, get_string=main_window.locale_manager.get_string)
        main_window.input_line.setText(main_window.settings.paths.input_dir)
        main_window.input_line.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Drag_Drop_Folder_Placeholder"))
        
        browse_button = QPushButton(main_window.locale_manager.get_string("MainWindow", "Browse_Button"))
        browse_button.clicked.connect(main_window.browse_folder)
        
        main_window.grid_view_button = QPushButton("3x3 edit")
        main_window.grid_view_button.setToolTip(main_window.locale_manager.get_string("MainWindow", "Switch_To_Grid_View"))
        main_window.grid_view_button.clicked.connect(main_window._show_grid_view)  # type: ignore

        main_window.model_combo = QComboBox()
        current_model_id = main_window.settings.model.model_id
        selected_index = 0
        selected_description = ""
        for i, entry in enumerate(model_registry.discover_models()):
            main_window.model_combo.addItem(entry.model_name, entry.model_id)
            description = main_window.locale_manager.get_string("ModelDescriptions", entry.model_id)
            main_window.model_combo.setItemData(i, description, Qt.ItemDataRole.ToolTipRole)
            if entry.model_id == current_model_id:
                selected_index = i
                selected_description = description
        main_window.model_combo.setCurrentIndex(selected_index)
        # The combo's own tooltip (shown while the dropdown is closed) tracks whichever
        # model is currently selected; _on_model_combo_changed() keeps it in sync.
        main_window.model_combo.setToolTip(selected_description)

        controls_layout.addWidget(main_window.input_line)
        controls_layout.addWidget(browse_button)
        controls_layout.addWidget(main_window.grid_view_button)
        controls_layout.addWidget(main_window.model_combo)
        layout.addLayout(controls_layout)

        # VLM 接続トグル（260901_VLM_*）。選択中モデル種別とは独立。ON のとき、
        # ローカルモデルの代わりにネットワーク VLM で生成する。出力形式（タグ列 /
        # 自然文）は VLM プロンプト次第なので、タグ付け・キャプションどちらにも使える。
        vlm_row = QHBoxLayout()
        main_window.use_vlm_check = QCheckBox(
            main_window.locale_manager.get_string("Vlm", "Use_Vlm_Checkbox"))
        main_window.use_vlm_check.setToolTip(
            main_window.locale_manager.get_string("Vlm", "Use_Vlm_Tooltip"))
        main_window.use_vlm_check.setChecked(bool(getattr(main_window.settings.vlm, "enabled", False)))
        main_window.use_vlm_check.toggled.connect(main_window._on_use_vlm_toggled)  # type: ignore
        main_window.vlm_settings_button = QPushButton(
            main_window.locale_manager.get_string("Vlm", "Method_Settings_Button"))
        main_window.vlm_single_test_button = QPushButton(
            main_window.locale_manager.get_string("Vlm", "Method_Single_Test_Button"))
        main_window.vlm_settings_button.clicked.connect(main_window._open_vlm_settings)  # type: ignore
        main_window.vlm_single_test_button.clicked.connect(main_window._run_vlm_single_test)  # type: ignore
        _vlm_on = bool(getattr(main_window.settings.vlm, "enabled", False))
        main_window.vlm_settings_button.setVisible(_vlm_on)
        main_window.vlm_single_test_button.setVisible(_vlm_on)
        vlm_row.addWidget(main_window.use_vlm_check)
        vlm_row.addWidget(main_window.vlm_settings_button)
        vlm_row.addWidget(main_window.vlm_single_test_button)
        vlm_row.addStretch(1)
        layout.addLayout(vlm_row)
        return group

    def _create_viewer_group(self, main_window: 'MainWindow') -> QSplitter:
        """Creates the splitter for the image viewer and tag editor."""
        img_tag_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        main_window.image_label = ClickableLabel()
        main_window.image_label.doubleClicked.connect(main_window.show_enlarged_image)
        main_window.image_label.setMinimumSize(225, 225)
        main_window.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_window.image_label.setToolTip(main_window.locale_manager.get_string("MainWindow", "ImageViewer_Tooltip"))
        
        tag_panel_widget = self._create_single_image_tag_panel(main_window)
        
        img_tag_splitter.addWidget(main_window.image_label)
        img_tag_splitter.addWidget(tag_panel_widget)
        img_tag_splitter.setSizes([200, 300])
        img_tag_splitter.setStretchFactor(0, 1)
        img_tag_splitter.setStretchFactor(1, 1)
        return img_tag_splitter

    def _create_single_image_tag_panel(self, main_window: 'MainWindow') -> QWidget:
        """Creates the panel for viewing and editing tags of a single image."""
        tag_panel_widget = QWidget()
        tag_panel = QVBoxLayout(tag_panel_widget)
        tag_panel.setContentsMargins(0, 0, 0, 0)
        
        # Header with title and undo/redo buttons
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Image_Tags_Label")))
        header_layout.addStretch(1)
        
        # Undo button
        main_window.undo_button = QPushButton("↶ Undo")
        main_window.undo_button.setEnabled(False)
        main_window.undo_button.setMaximumWidth(80)
        main_window.undo_button.setToolTip(main_window.locale_manager.get_string("MainWindow", "Undo_No_Actions"))
        main_window.undo_button.clicked.connect(main_window._perform_undo)  # type: ignore
        header_layout.addWidget(main_window.undo_button)
        
        # Redo button
        main_window.redo_button = QPushButton("↷ Redo")
        main_window.redo_button.setEnabled(False)
        main_window.redo_button.setMaximumWidth(80)
        main_window.redo_button.setToolTip(main_window.locale_manager.get_string("MainWindow", "Redo_No_Actions"))
        main_window.redo_button.clicked.connect(main_window._perform_redo)  # type: ignore
        header_layout.addWidget(main_window.redo_button)
        
        tag_panel.addLayout(header_layout)
        
        main_window.tag_grid_container = QWidget()
        main_window.tag_display_grid = QGridLayout(main_window.tag_grid_container)
        main_window.tag_grid_container.setLayout(main_window.tag_display_grid)
        min_grid_height = main_window.settings.window.tag_display_rows * (main_window._tag_button_min_height + 5)  # type: ignore
        main_window.tag_grid_container.setMinimumHeight(min_grid_height)
        main_window.tag_grid_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tag_panel.addWidget(main_window.tag_grid_container)

        # Caption mode (Florence-2, model_type="captioner"): free-text editor shown instead
        # of the tag button grid above. Hidden by default; ModelModeController toggles
        # visibility based on the active model's model_type (design.md 8.5節).
        main_window.caption_text_edit = QTextEdit()
        main_window.caption_text_edit.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Caption_Placeholder"))
        main_window.caption_text_edit.setVisible(False)
        # captioner / VLM モードではタグボタン群が消えるので、この編集欄が余った縦領域を
        # すべて埋めるようにする（stretch=1）。tag_panel 末尾のばね（addStretch）は
        # ModelModeController が text_ui のとき stretch=0 に殺すので、編集欄の下に空きが
        # 出ず、長文でも不要なスクロールが発生しない。
        main_window.caption_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_window.caption_text_edit.setMinimumHeight(120)
        tag_panel.addWidget(main_window.caption_text_edit, 1)

        # 生成キャプションを既存 .txt にどう入れるか（前に追加 / 後に追加 / 上書き）。
        # 排他のトグルボタンで、押した1つだけが押しっぱなしになる。danbooru タグと
        # 自然言語を1ファイルに同居させるモデル向け（2026-08-31 ユーザー要望）。
        # 親を main_window にしておく（この後 shared_run_block へ addWidget で移すまでの間、
        # 親なしトップレベルにならないように）。
        main_window.caption_placement_widget = QWidget(main_window)
        placement_row = QHBoxLayout(main_window.caption_placement_widget)
        placement_row.setContentsMargins(0, 0, 0, 0)
        main_window.caption_placement_group = QButtonGroup(main_window)
        main_window.caption_placement_group.setExclusive(True)
        main_window.caption_placement_buttons = {}
        for key, label_key in (("PREPEND", "Caption_Placement_Prepend"),
                               ("APPEND", "Caption_Placement_Append"),
                               ("OVERWRITE", "Caption_Placement_Overwrite")):
            btn = QPushButton(main_window.locale_manager.get_string("MainWindow", label_key))
            btn.setCheckable(True)
            btn.setToolTip(main_window.locale_manager.get_string("MainWindow", f"{label_key}_Tooltip"))
            btn.setProperty("placement", key)
            main_window.caption_placement_group.addButton(btn)
            main_window.caption_placement_buttons[key] = btn
            placement_row.addWidget(btn)
        placement_row.addStretch(1)
        current_placement = str(main_window.settings.caption.placement).upper()
        main_window.caption_placement_buttons.get(
            current_placement, main_window.caption_placement_buttons["OVERWRITE"]).setChecked(True)
        main_window.caption_placement_group.buttonClicked.connect(main_window._on_caption_placement_changed)  # type: ignore
        main_window.caption_placement_widget.setVisible(False)
        # NOTE: caption_placement_widget は viewer ではなく section[1] の shared_run_block へ
        # 載せる（_create_bulk_actions_group）。Run ボタン直上に「出力の入れ方」系をまとめ、
        # tagger/captioner/VLM のどのモードでも同じ位置に出すため。ここでは生成だけ。

        main_window.task_combo = QComboBox()
        main_window.task_combo.setToolTip(main_window.locale_manager.get_string("MainWindow", "Task_Combo_Tooltip"))
        # "More Detailed Caption" first: it's the usual choice, and it becomes the
        # fallback when settings.caption.task is unset/unknown (findData -> -1 -> index 0).
        for task_key in ("MORE_DETAILED_CAPTION", "DETAILED_CAPTION", "CAPTION"):
            main_window.task_combo.addItem(
                main_window.locale_manager.get_string("MainWindow", f"Task_Label_{task_key}"), task_key)
        current_task_index = max(0, main_window.task_combo.findData(main_window.settings.caption.task))
        main_window.task_combo.setCurrentIndex(current_task_index)
        main_window.task_combo.setVisible(False)
        tag_panel.addWidget(main_window.task_combo)

        page_nav_layout = QHBoxLayout()
        main_window.image_tag_prev_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Previous_Page"))
        main_window.image_tag_prev_page_btn.clicked.connect(lambda: main_window._change_image_tag_page(-1))  # type: ignore
        main_window.image_tag_next_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Next_Page"))
        main_window.image_tag_next_page_btn.clicked.connect(lambda: main_window._change_image_tag_page(1))  # type: ignore
        page_nav_layout.addWidget(main_window.image_tag_prev_page_btn)
        page_nav_layout.addWidget(main_window.image_tag_next_page_btn)
        tag_panel.addLayout(page_nav_layout)

        main_window.add_single_tag_label = QLabel(main_window.locale_manager.get_string("MainWindow", "Add_Single_Tag_Label"))
        tag_panel.addWidget(main_window.add_single_tag_label)
        add_tag_layout = QHBoxLayout()
        main_window.add_single_tag_line = QLineEdit()
        main_window.add_single_tag_line.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Tags_Comma_Separated_Placeholder"))
        main_window.add_single_tag_line.setToolTip(main_window.locale_manager.get_string("MainWindow", "AddTag_Hover_Tooltip"))
        main_window.add_single_tag_line.returnPressed.connect(main_window._add_single_tag)  # type: ignore
        main_window.add_single_tag_button = QPushButton(main_window.locale_manager.get_string("MainWindow", "Add_Button"))
        main_window.add_single_tag_button.clicked.connect(main_window._add_single_tag)  # type: ignore
        add_tag_layout.addWidget(main_window.add_single_tag_line)
        add_tag_layout.addWidget(main_window.add_single_tag_button)
        tag_panel.addLayout(add_tag_layout)

        # tagger モードでは caption_text_edit が隠れて stretch が効かないので、この末尾の
        # ばねで下端の余白を保持し、タグ関連ウィジェットを上詰めにする。caption / VLM
        # モードでは ModelModeController がこのばねを stretch=0 にして殺し、
        # caption_text_edit(stretch=1) が編集欄を縦いっぱいへ伸ばす。
        tag_panel.addStretch(1)
        main_window._tag_panel_layout = tag_panel
        main_window._tag_panel_bottom_spacer_index = tag_panel.count() - 1
        return tag_panel_widget

    def _create_bulk_actions_group(self, main_window: 'MainWindow') -> QWidget:
        """右スプリッター section[1]。

        - ``tagger_config_block``: 一括削除／一括追加＋しきい値・最大タグ数スライダー。
          ONNX タガー専用なので captioner / VLM 接続時は丸ごと非表示（ModelModeController）。
        - ``shared_run_block``: 生成キャプションの挿入位置・既存ファイルモード・対象モード・
          Run ボタン。どのモードでも表示し、Run 直上へ集約する。
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- tagger 専用ブロック（captioner / VLM では非表示） ---
        main_window.tagger_config_block = QWidget()
        tcb_layout = QVBoxLayout(main_window.tagger_config_block)
        tcb_layout.setContentsMargins(0, 0, 0, 0)

        bulk_edit_layout = QHBoxLayout()
        main_window.bulk_delete_group = self._create_bulk_delete_group(main_window)
        main_window.bulk_add_group = self._create_bulk_add_group(main_window)
        bulk_edit_layout.addWidget(main_window.bulk_delete_group, 3)
        bulk_edit_layout.addWidget(main_window.bulk_add_group, 1)
        tcb_layout.addLayout(bulk_edit_layout)
        tcb_layout.addWidget(self._create_tagger_sliders(main_window))

        layout.addWidget(main_window.tagger_config_block)

        # --- 共通ブロック（常時表示） ---
        main_window.shared_run_block = QWidget()
        srb_layout = QVBoxLayout(main_window.shared_run_block)
        srb_layout.setContentsMargins(0, 0, 0, 0)
        srb_layout.addWidget(main_window.caption_placement_widget)
        srb_layout.addLayout(self._create_mode_row(main_window))

        main_window.run_button = QPushButton(main_window.locale_manager.get_string("Constants", "Tag_Button_Text"))
        main_window.run_button.setStyleSheet(constants.STYLE_BTN_GREEN)
        srb_layout.addWidget(main_window.run_button)

        layout.addWidget(main_window.shared_run_block)
        return widget

    def _create_bulk_delete_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the group for bulk tag deletion."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Bulk_Delete_Tags"))
        layout = QVBoxLayout(group)
        
        header_layout = QHBoxLayout()
        main_window.loading_label = QLabel(main_window.locale_manager.get_string("Constants", "Loading_Tag_List"))
        main_window.loading_label.setStyleSheet("font-weight:bold;")
        header_layout.addWidget(main_window.loading_label)

        header_layout.addStretch(1)

        main_window.language_combo = QComboBox()
        main_window.language_combo.addItems([
            "English", 
            "日本語", 
            "Français", 
            "Deutsch", 
            "Español", 
            "Русский", 
            "简体中文", 
            "繁體中文", 
            "한국어"
        ])
        main_window.language_combo.setCurrentIndex(0)
        header_layout.addWidget(main_window.language_combo)
        
        layout.addLayout(header_layout)
        
        main_window.tag_button_grid = QGridLayout()
        for i in range(4):
            main_window.tag_button_grid.setColumnStretch(i, 1)
        layout.addLayout(main_window.tag_button_grid)
        
        page_nav_layout = QHBoxLayout()
        main_window.prev_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Previous_16_Items"))
        main_window.prev_page_btn.clicked.connect(lambda: main_window._change_tag_page(-1))  # type: ignore
        main_window.next_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Next_16_Items"))
        main_window.next_page_btn.clicked.connect(lambda: main_window._change_tag_page(1))  # type: ignore
        page_nav_layout.addWidget(main_window.prev_page_btn)
        page_nav_layout.addWidget(main_window.next_page_btn)
        layout.addLayout(page_nav_layout)
        
        return group

    def _create_bulk_add_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the group for bulk tag addition."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Bulk_Add_Tags"))
        layout = QVBoxLayout(group)
        
        layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Files")))
        main_window.add_tag_line = QLineEdit()
        main_window.add_tag_line.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Placeholder"))
        main_window.add_tag_line.setToolTip(main_window.locale_manager.get_string("MainWindow", "Comma_Recommended_Tooltip"))
        main_window.add_tag_button = QPushButton(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Txt_Files"))
        main_window.add_tag_button.setStyleSheet(constants.STYLE_BTN_BLUE)
        main_window.add_tag_button.clicked.connect(lambda: main_window.add_tag_all(prepend=True))
        layout.addWidget(main_window.add_tag_line)
        layout.addWidget(main_window.add_tag_button)

        layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Files_Append")))
        main_window.add_tag_line_append = QLineEdit()
        main_window.add_tag_line_append.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Placeholder"))
        main_window.add_tag_line_append.setToolTip(main_window.locale_manager.get_string("MainWindow", "Comma_Recommended_Tooltip"))
        main_window.add_tag_button_append = QPushButton(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Txt_Files_Append"))
        main_window.add_tag_button_append.setStyleSheet(constants.STYLE_BTN_BLUE)
        main_window.add_tag_button_append.clicked.connect(lambda: main_window.add_tag_all(prepend=False))
        layout.addWidget(main_window.add_tag_line_append)
        layout.addWidget(main_window.add_tag_button_append)

        layout.addStretch(1)
        return group

    def _create_tagger_sliders(self, main_window: 'MainWindow') -> QWidget:
        """しきい値・最大タグ数スライダーとカテゴリ設定ボタンの行（ONNX タガー専用）。"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        thresh_group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Tag_Threshold"))
        thresh_layout = QGridLayout(thresh_group)
        main_window.create_slider_group(thresh_layout, 'Thresholds', 0.0, 1.0, 0.01, {'general': 0, 'character': 1})

        limit_group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Max_Tags"))
        limit_layout = QGridLayout(limit_group)
        # Minimum is 0, not 1: inference treats a 0 max-tag count as "emit nothing from
        # this category", and the per-category dialog offers 0. With a minimum of 1 the
        # slider silently clamped a stored 0 back up to 1 when the two were synced.
        main_window.create_slider_group(limit_layout, 'Limits', 0, 150, 1, {'general': 0})
        main_window.create_slider_group(limit_layout, 'Limits', 0, 10, 1, {'character': 1})

        # Opens the per-category (rating / copyright / artist / meta ...) threshold &
        # max-tag dialog. Per user request (2026-08-31, after several layout attempts):
        # a full-row-height button at the right with a matching larger font, so it fills
        # the vertical space instead of leaving a gap under a small button.
        main_window.category_settings_button = QPushButton(
            main_window.locale_manager.get_string("MainWindow", "Category_Settings_Button"))
        main_window.category_settings_button.setToolTip(
            main_window.locale_manager.get_string("MainWindow", "Category_Settings_Tooltip"))
        main_window.category_settings_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        main_window.category_settings_button.setStyleSheet("font-size: 12pt; padding: 4px 10px;")
        main_window.category_settings_button.clicked.connect(main_window._open_category_settings_dialog)

        layout.addWidget(thresh_group, 1)
        layout.addWidget(limit_group, 1)
        layout.addWidget(main_window.category_settings_button, 0)
        return widget

    def _create_mode_row(self, main_window: 'MainWindow') -> QHBoxLayout:
        """既存ファイルの扱い（ASK / OVERWRITE / SKIP / APPEND）と一括処理の対象選択
        （すべて / 未生成のみ / 失敗のみ / 選択画像のみ）を Run ボタン直上へ並べる行。
        tagger / captioner / VLM のどのモードでも意味があるので常時表示（spec.md 6.1節・
        260903_vlm-gap-fix.md todo 5）。"""
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Existing_Mode_Label")))
        main_window.existing_mode_combo = QComboBox()
        for key, label_key in (("ASK", "Existing_Mode_Ask"), ("OVERWRITE", "Existing_Mode_Overwrite"),
                               ("SKIP", "Existing_Mode_Skip"), ("APPEND", "Existing_Mode_Append")):
            main_window.existing_mode_combo.addItem(
                main_window.locale_manager.get_string("MainWindow", label_key), key)
        current = str(main_window.settings.behavior.existing_file_mode).upper()
        idx = main_window.existing_mode_combo.findData(current)
        main_window.existing_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        main_window.existing_mode_combo.currentIndexChanged.connect(main_window._on_existing_mode_changed)  # type: ignore
        mode_row.addWidget(main_window.existing_mode_combo)

        mode_row.addSpacing(12)
        mode_row.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Target_Mode_Label")))
        main_window.target_mode_combo = QComboBox()
        for key, label_key in (("ALL", "Target_Mode_All"), ("UNPROCESSED", "Target_Mode_Unprocessed"),
                               ("FAILED", "Target_Mode_Failed"), ("SELECTED", "Target_Mode_Selected")):
            main_window.target_mode_combo.addItem(
                main_window.locale_manager.get_string("MainWindow", label_key), key)
        _tm_current = str(main_window.settings.behavior.target_mode).upper()
        _tm_idx = main_window.target_mode_combo.findData(_tm_current)
        main_window.target_mode_combo.setCurrentIndex(_tm_idx if _tm_idx >= 0 else 0)
        main_window.target_mode_combo.currentIndexChanged.connect(main_window._on_target_mode_changed)  # type: ignore
        mode_row.addWidget(main_window.target_mode_combo)

        mode_row.addStretch(1)
        return mode_row

    def _create_log_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the group for the execution log."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Execution_Log"), main_window)
        layout = QVBoxLayout(group)
        main_window.log_output = QTextEdit()
        main_window.log_output.setReadOnly(True)
        main_window.log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(main_window.log_output)
        main_window.log_output.setMinimumHeight(100)
        main_window.log_output.setMaximumHeight(400)
        return group
