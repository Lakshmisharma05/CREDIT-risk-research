import sys
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QStackedWidget, QTableWidget, QTableWidgetItem,
    QScrollArea, QAbstractItemView, QLineEdit
)
from PySide6.QtCore import Qt


class CredLensApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CREDLENS AI | Credit Intelligence")
        self.resize(1250, 800)
        self.setMinimumSize(1000, 650)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0B1220;
                color: #F8FAFC;
                font-family: Arial;
            }
            QLabel {
                background-color: transparent;
            }
            QFrame#sidebar {
                background-color: #111C2E;
                border-right: 1px solid #26364D;
            }
            QLabel#logo {
                color: #60A5FA;
                font-size: 25px;
                font-weight: bold;
            }
            QLabel#subtitle {
                color: #94A3B8;
                font-size: 11px;
            }
            QPushButton {
                background-color: transparent;
                color: #CBD5E1;
                border: none;
                padding: 13px;
                text-align: left;
                font-size: 14px;
                border-radius: 7px;
            }
            QPushButton:hover {
                background-color: #233550;
                color: white;
            }
            QPushButton:checked {
                background-color: #1D4ED8;
                color: white;
            }
            QFrame#panel {
                background-color: #17243A;
                border: 1px solid #26364D;
                border-radius: 12px;
            }
            QLabel#heading {
                font-size: 27px;
                font-weight: bold;
            }
            QLabel#pageTitle {
                font-size: 18px;
                font-weight: bold;
                margin-top: 10px;
            }
            QLabel#muted {
                color: #94A3B8;
                font-size: 13px;
            }
            QLabel#card_title {
                color: #94A3B8;
                font-size: 13px;
            }
            QLabel#card_value {
                color: #F8FAFC;
                font-size: 22px;
                font-weight: bold;
            }
            QTableWidget {
                background-color: #17243A;
                gridline-color: #26364D;
                border: 1px solid #26364D;
            }
            QHeaderView::section {
                background-color: #111C2E;
                color: #CBD5E1;
                padding: 6px;
                border: 1px solid #26364D;
            }
            QScrollArea {
                border: none;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(235)

        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 25, 18, 20)

        logo = QLabel("CREDLENS AI")
        logo.setObjectName("logo")

        subtitle = QLabel("CREDIT INTELLIGENCE")
        subtitle.setObjectName("subtitle")

        side_layout.addWidget(logo)
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(35)

        self.stack = QStackedWidget()
        self.buttons = []

        pages = [
            "Dashboard",
            "Data Intelligence",
            "Risk Analysis",
            "Explainability Lab",
            "Simulation Lab",
            "Fairness & Reliability",
            "Report Studio"
        ]

        for index, page_name in enumerate(pages):
            button = QPushButton(page_name)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, i=index: self.change_page(i)
            )

            side_layout.addWidget(button)
            self.buttons.append(button)

            # Wrap each page in a scroll area so tall pages don't get squashed
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.create_page(page_name))
            self.stack.addWidget(scroll)

        self.buttons[0].setChecked(True)
        side_layout.addStretch()

        version = QLabel("CREDLENS AI  •  v1.0")
        version.setObjectName("subtitle")
        side_layout.addWidget(version)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack)

    def create_card(self, title, value, description=""):
        card = QFrame()
        card.setObjectName("panel")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("card_title")

        value_label = QLabel(value)
        value_label.setObjectName("card_value")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        if description:
            description_label = QLabel(description)
            description_label.setObjectName("muted")
            layout.addWidget(description_label)

        layout.addStretch()

        return card

    def make_table(self, headers, rows, max_height):
        table = QTableWidget()
        table.setRowCount(len(rows))
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                table.setItem(r, c, QTableWidgetItem(text))

        table.setMinimumHeight(min(max_height, 40 + 30 * len(rows)))
        table.setMaximumHeight(max_height)
        return table

    def create_page(self, page_name):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(35, 30, 35, 30)
        layout.setSpacing(18)

        heading = QLabel(page_name)
        heading.setObjectName("heading")

        description = QLabel(
            "Explainable Credit Intelligence & Risk Simulation"
        )
        description.setObjectName("muted")

        layout.addWidget(heading)
        layout.addWidget(description)

        if page_name == "Dashboard":
            welcome = QLabel("Welcome to CREDLENS AI")
            welcome.setStyleSheet(
                "font-size: 23px; font-weight: bold; color: #F8FAFC;"
            )

            intro = QLabel(
                "Your intelligent workspace for understanding "
                "credit risk and model decisions."
            )
            intro.setObjectName("muted")

            layout.addSpacing(12)
            layout.addWidget(welcome)
            layout.addWidget(intro)
            layout.addSpacing(15)
            try:
                csv_path = Path(__file__).parent.parent / "cs-training.csv"
                df = pd.read_csv(csv_path)

                dataset_status = "Connected"
                dataset_description = "Dataset successfully loaded"

                total_records = f"{len(df):,}"
                total_features = str(len(df.columns))
                missing_values = f"{int(df.isnull().sum().sum()):,}"
                duplicate_records = f"{int(df.duplicated().sum()):,}"
            except Exception as e:
                dataset_status = "Unavailable"
                dataset_description = str(e)
                print("DASHBOARD ERROR:", repr(e))

                total_records = "N/A"
                total_features = "N/A"
                missing_values = "N/A"
                duplicate_records = "N/A"

            try:
                prediction_path = (
                    Path(__file__).parent.parent
                    / "modeling" / "results"
                    / "customer_predictions.csv"
                )
                predictions = pd.read_csv(prediction_path)

                low_risk = str((predictions["risk_band"] == "Low").sum())
                medium_risk = str((predictions["risk_band"] == "Medium").sum())
                high_risk = str((predictions["risk_band"] == "High").sum())
                avg_probability = f"{predictions['probability'].mean():.1%}"

            except Exception as e:
                print("PREDICTION ERROR:", repr(e))
                low_risk = medium_risk = high_risk = "N/A"
                avg_probability = "N/A"

            grid = QGridLayout()
            grid.setSpacing(16)

            grid.addWidget(
                self.create_card(
                    "TOTAL RECORDS",
                    total_records,
                    "Records available in dataset"
                ), 0, 0
            )

            grid.addWidget(
                self.create_card(
                    "TOTAL FEATURES",
                    total_features,
                    "Dataset columns"
                ), 0, 1
            )

            grid.addWidget(
                self.create_card(
                    "MISSING VALUES",
                    missing_values,
                    "Values requiring data quality review"
                ), 1, 0
            )

            grid.addWidget(
                self.create_card(
                    "DUPLICATE RECORDS",
                    duplicate_records,
                    "Repeated rows in dataset"
                ), 1, 1
            )
            grid.addWidget(
                self.create_card("LOW RISK", low_risk, "Low risk customers"), 2, 0
            )
            grid.addWidget(
                self.create_card("MEDIUM RISK", medium_risk, "Medium risk customers"), 2, 1
            )
            grid.addWidget(
                self.create_card("HIGH RISK", high_risk, "High risk customers"), 3, 0
            )
            grid.addWidget(
                self.create_card("AVG DEFAULT PROBABILITY", avg_probability, "Average predicted probability"), 3, 1
            )
            layout.addLayout(grid)
            layout.addSpacing(15)

            activity = QFrame()
            activity.setObjectName("panel")

            activity_layout = QVBoxLayout(activity)
            activity_layout.setContentsMargins(20, 18, 20, 18)

            activity_title = QLabel("PROJECT WORKSPACE")
            activity_title.setStyleSheet(
                "font-size: 16px; font-weight: bold;"
            )

            activity_text = QLabel(
                "CREDLENS AI is ready for credit risk analysis.\n\n"
                "Dataset Status: Successfully Connected\n"
                "Data Intelligence: EDA and Outlier Analysis Available\n"
                "Model Integration: Successfully Connected\n"
                "Explainability: SHAP integration pending\n\n"
                "Explore the modules to understand dataset quality, "
                "feature patterns and credit risk insights."
            )
            activity_text.setObjectName("muted")

            activity_layout.addWidget(activity_title)
            activity_layout.addSpacing(10)
            activity_layout.addWidget(activity_text)

            layout.addWidget(activity)
            layout.addStretch()

        elif page_name == "Data Intelligence":
            description.setText("Dataset Overview & Data Quality Analysis")
            layout.addSpacing(25)

            try:
                csv_path = Path(__file__).parent.parent / "cs-training.csv"
                df = pd.read_csv(csv_path)

                info = [
                    ("TOTAL RECORDS", f"{df.shape[0]:,}"),
                    ("TOTAL FEATURES", str(df.shape[1])),
                    ("MISSING VALUES", f"{int(df.isnull().sum().sum()):,}"),
                    ("DUPLICATE RECORDS", f"{int(df.duplicated().sum()):,}")
                ]

                grid = QGridLayout()
                grid.setSpacing(16)

                for i, (label_text, value) in enumerate(info):
                    grid.addWidget(
                        self.create_card(label_text, value),
                        i // 2, i % 2
                    )

                layout.addLayout(grid)

                # ---- Dataset preview ----
                preview_title = QLabel("Dataset Preview")
                preview_title.setObjectName("pageTitle")
                layout.addWidget(preview_title)

                preview = df.head(10)
                layout.addWidget(
                    self.make_table(
                        [str(c) for c in df.columns],
                        preview.values.tolist(),
                        350
                    )
                )

                # ---- Data quality table ----
                quality_title = QLabel("Data Quality Analysis")
                quality_title.setObjectName("pageTitle")
                layout.addWidget(quality_title)

                missing_df = pd.DataFrame({
                    "Feature": df.columns,
                    "Missing Values": df.isnull().sum().values,
                    "Missing (%)": (df.isnull().mean().values * 100).round(2)
                })

                layout.addWidget(
                    self.make_table(
                        ["Feature", "Missing Values", "Missing (%)"],
                        missing_df.values.tolist(),
                        350
                    )
                )

                # ---- Missing values chart ----
                chart_title = QLabel("Missing Values Distribution")
                chart_title.setObjectName("pageTitle")
                layout.addWidget(chart_title)

                figure = Figure(figsize=(8, 4), facecolor="#17243A")
                canvas = FigureCanvas(figure)
                ax = figure.add_subplot(111)
                ax.set_facecolor("#17243A")

                missing_counts = df.isnull().sum()
                missing_counts = missing_counts[missing_counts > 0]

                if missing_counts.empty:
                    ax.text(0.5, 0.5, "No missing values",
                            ha="center", va="center", color="#F8FAFC",
                            transform=ax.transAxes)
                    ax.set_axis_off()
                else:
                    ax.barh(
                        [str(i) for i in missing_counts.index],
                        missing_counts.values,
                        color="#60A5FA"
                    )
                    ax.set_xlabel("Number of Missing Values", color="#CBD5E1")
                    ax.set_ylabel("Features", color="#CBD5E1")
                    ax.set_title("Feature-wise Missing Values", color="#F8FAFC")
                    ax.tick_params(colors="#CBD5E1")
                    for spine in ax.spines.values():
                        spine.set_color("#26364D")
                    ax.invert_yaxis()

                figure.tight_layout()

                canvas.setMinimumHeight(350)
                layout.addWidget(canvas)
                feature_title = QLabel("Feature Distribution Analysis")
                feature_title.setObjectName("heading")
                layout.addWidget(feature_title)
                features = [
                    "age",
                    "MonthlyIncome",
                    "DebtRatio",
                    "RevolvingUtilizationOfUnsecuredLines"
                ]
                figure2 = Figure(figsize=(10, 7))
                canvas2 = FigureCanvas(figure2)
                for i, feature in enumerate(features):
                    ax = figure2.add_subplot(2, 2, i + 1)
                    ax.hist(
                        df[feature].dropna(),
                        bins=30,
                        edgecolor="white"
                    )
                    ax.set_title(feature)
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Frequency")
                figure2.tight_layout()
                canvas2.setMinimumHeight(550)
                layout.addWidget(canvas2)
                # correlation analysis
                correlation_title = QLabel("Correlation Analysis")
                correlation_title.setObjectName("heading")
                layout.addWidget(correlation_title)
                corr = df.corr(numeric_only=True)
                figure3 = Figure(figsize=(10, 8))
                canvas3 = FigureCanvas(figure3)
                ax = figure3.add_subplot(111)
                image = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
                ax.set_yticklabels(corr.columns, fontsize=7)
                ax.set_title("Feature Correlation Heatmap")
                figure3.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                figure3.tight_layout()
                canvas3.setMinimumHeight(650)
                layout.addWidget(canvas3)
                outlier_title = QLabel("Outlier Analysis")
                outlier_title.setObjectName("heading")
                layout.addWidget(outlier_title)
                numerical_df = df.select_dtypes(include="number")
                outlier_rows = []
                for column in numerical_df.columns:
                    Q1 = numerical_df[column].quantile(0.25)
                    Q3 = numerical_df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    count = ((numerical_df[column] < lower) | (numerical_df[column] > upper)).sum()
                    percentage = (count / len(numerical_df)) * 100
                    outlier_rows.append([column, int(count), round(percentage, 2)])
                layout.addWidget(
                    self.make_table(
                        ["Feature", "Outlier Count", "Outlier (%)"],
                        outlier_rows,
                        400
                    )
                )
                outlier_chart_title = QLabel("Outlier Distribution")
                outlier_chart_title.setObjectName("pageTitle")
                layout.addWidget(outlier_chart_title)
                figure4 = Figure(figsize=(10, 5))
                canvas4 = FigureCanvas(figure4)
                ax = figure4.add_subplot(111)
                chart_data = [row for row in outlier_rows if row[1] > 0]
                if chart_data:
                    ax.barh(
                        [row[0] for row in chart_data],
                        [row[1] for row in chart_data],
                        color="#F59E0B"
                    )
                    ax.set_xlabel("Outlier Count")
                    ax.set_title("Feature-wise Outlier Distribution")
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, "No outliers detected",
                            ha="center", va="center",
                            transform=ax.transAxes
                    )
                    ax.set_axis_off()
                figure4.tight_layout()
                canvas4.setMinimumHeight(400)
                layout.addWidget(canvas4)
            except Exception as e:
                error = QLabel(f"Dataset Error: {e}")
                error.setObjectName("muted")
                error.setWordWrap(True)
                layout.addWidget(error)

            layout.addStretch()

        elif page_name == "Risk Analysis":
            description.setText("Borrower Risk Overview & Credit Risk Insights")
            layout.addSpacing(20)

            try:
                csv_path = Path(__file__).parent.parent / "cs-training.csv"
                df = pd.read_csv(csv_path)

                target = "SeriousDlqin2yrs"

                if target in df.columns:
                    default_count = int(df[target].sum())
                    total = len(df)
                    non_default = total - default_count
                    default_rate = (default_count / total) * 100 if total else 0

                    info = [
                        ("TOTAL BORROWERS", f"{total:,}"),
                        ("HIGH-RISK CASES", f"{default_count:,}"),
                        ("NON-DEFAULT CASES", f"{non_default:,}"),
                        ("DEFAULT RATE", f"{default_rate:.2f}%")
                    ]

                    grid = QGridLayout()
                    grid.setSpacing(16)

                    for i, (label_text, value) in enumerate(info):
                        grid.addWidget(
                            self.create_card(label_text, value),
                            i // 2, i % 2
                        )

                    layout.addLayout(grid)

                    risk_title = QLabel("Credit Risk Distribution")
                    risk_title.setObjectName("pageTitle")
                    layout.addWidget(risk_title)

                    figure = Figure(figsize=(8, 4))
                    canvas = FigureCanvas(figure)
                    ax = figure.add_subplot(111)

                    ax.bar(
                        ["Non-Default", "Default"],
                        [non_default, default_count],
                        color=["#60A5FA", "#F87171"]
                    )
                    ax.set_ylabel("Number of Borrowers")
                    ax.set_title("Borrower Default Distribution")

                    figure.tight_layout()
                    canvas.setMinimumHeight(350)
                    layout.addWidget(canvas)

                    note = QLabel(
                        "The chart above shows actual historical default labels. "
                        "The section below shows predictions generated by the trained model. "
                    )
                    note.setObjectName("muted")
                    note.setWordWrap(True)
                    layout.addWidget(note)
                    prediction_path = (
                        Path(__file__).parent.parent
                        / "modeling" / "results"
                        / "customer_predictions.csv"
                    )
                    predictions = pd.read_csv(prediction_path)
                    low = int((predictions["risk_band"] == "Low").sum())
                    medium = int((predictions["risk_band"] == "Medium").sum())
                    high = int((predictions["risk_band"] == "High").sum())
                    avg_probability = predictions['probability'].mean()
                    model_title = QLabel("Model-Based Risk Distribution")
                    model_title.setObjectName("pageTitle")
                    layout.addWidget(model_title)
                    model_grid = QGridLayout()
                    model_grid.setSpacing(16)
                    model_grid.addWidget(
                        self.create_card("LOW RISK", f"{low:,}"), 0, 0
                    )
                    model_grid.addWidget(
                        self.create_card("MEDIUM RISK", f"{medium:,}"), 0, 1
                    )
                    model_grid.addWidget(
                        self.create_card("HIGH RISK", f"{high:,}"), 1, 0
                    )
                    model_grid.addWidget(
                        self.create_card("AVG DEFAULT PROBABILITY", f"{avg_probability:.1%}"), 1, 1
                    )
                    layout.addLayout(model_grid)
                    model_figure = Figure(figsize=(8, 4))
                    model_canvas = FigureCanvas(model_figure)
                    model_ax = model_figure.add_subplot(111)
                    model_ax.bar(
                        ["Low Risk", "Medium Risk", "High Risk"],
                        [low, medium, high],
                        color=["#60A5FA", "#FBBF24", "#F87171"]
                    )
                    model_ax.set_ylabel("Number of Borrowers")
                    model_ax.set_title("Predicted Risk Distribution")
                    model_figure.tight_layout()
                    model_canvas.setMinimumHeight(350)
                    layout.addWidget(model_canvas)

                else:
                    layout.addWidget(QLabel(
                        "Target column SeriousDlqin2yrs not found."
                    ))

            except Exception as e:
                error = QLabel(f"Risk Analysis Error: {e}")
                error.setWordWrap(True)
                layout.addWidget(error)

        elif page_name == "Explainability Lab":
            description.setText("SHAP-Based Model Explainability")
            layout.addSpacing(20)

            title = QLabel("Individual Prediction Explanation")
            title.setObjectName("pageTitle")
            layout.addWidget(title)

            note = QLabel(
                "Explore how borrower features influence the model's "
                "predicted default probability using SHAP values."
            )
            note.setObjectName("muted")
            note.setWordWrap(True)
            layout.addWidget(note)

            explain_button = QPushButton("GENERATE SHAP EXPLANATION")
            layout.addWidget(explain_button)

            result_label = QLabel(
                "Click the button to generate an explanation."
            )
            result_label.setWordWrap(True)
            layout.addWidget(result_label)

            figure = Figure(figsize=(8, 5))
            canvas = FigureCanvas(figure)
            canvas.setMinimumHeight(400)
            layout.addWidget(canvas)

            def generate_explanation():
                try:
                    print("EXPLAINABILITY ROOT:", Path(__file__).resolve().parent.parent)
                    import shap
                    project_root = Path(__file__).resolve().parent.parent
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    from modeling.credlens_package.credlens_predictor import (
                        CredLensPredictor, prepare_raw,
                        FEATURES, RAW_COLUMNS, to_logit
                    )

                    csv_path = (
                        Path(__file__).parent.parent / "cs-training.csv"
                    )
                    df = pd.read_csv(csv_path).drop(
                        columns=["Unnamed: 0"], errors="ignore"
                    )

                    p = CredLensPredictor()

                    sample_raw = df[RAW_COLUMNS].iloc[[0]].copy()

                    X = prepare_raw(
                        df[RAW_COLUMNS].sample(100, random_state=42)
                    )[FEATURES]

                    def predict_calibrated(data):
                        data = pd.DataFrame(data, columns=FEATURES)
                        raw = p.model.predict_proba(data)[:, 1]
                        return p.calibrator.predict_proba(
                            to_logit(raw)
                        )[:, 1]

                    explainer = shap.PermutationExplainer(
                        predict_calibrated,
                        X.sample(10, random_state=42)
                    )

                    sample = prepare_raw(sample_raw)[FEATURES]

                    explanation = explainer(
                        sample, max_evals=25
                    )

                    prediction = p.predict_df(sample_raw).iloc[0]

                    values = explanation.values[0]

                    importance = pd.Series(
                        values, index=FEATURES
                    ).sort_values(
                        key=abs, ascending=True
                    ).tail(10)

                    figure.clear()
                    ax = figure.add_subplot(111)

                    colors = [
                        "#F87171" if value > 0 else "#60A5FA"
                        for value in importance.values
                    ]

                    ax.barh(
                        importance.index,
                        importance.values,
                        color=colors
                    )

                    ax.axvline(0, color="gray", linewidth=0.8)
                    ax.set_xlabel(
                        "SHAP impact on default probability"
                    )
                    ax.set_title(
                        "Top 10 SHAP Feature Contributions"
                    )

                    figure.tight_layout()
                    canvas.draw()

                    result_label.setText(
                        f"Prediction: {prediction['label']}\n"
                        f"Risk Category: {prediction['risk_band']}\n"
                        f"Default Probability: "
                        f"{prediction['probability']:.1%}\n"
                        f"Model Version: {prediction['model_version']}\n\n"
                        "Red = increases predicted default probability. "
                        "Blue = decreases predicted default probability."
                    )

                except Exception as e:
                    print("FULL EXPLAINABILITY ERROR:", repr(e))
                    result_label.setText(
                        f"Explainability Error: {e}"
                    )

            explain_button.clicked.connect(generate_explanation)
            layout.addStretch()

        elif page_name == "Simulation Lab":
            description.setText("Interactive Credit Risk Simulation")
            layout.addSpacing(20)

            title = QLabel("Borrower Financial Profile")
            title.setObjectName("pageTitle")
            layout.addWidget(title)

            income_label = QLabel("Monthly Income")
            layout.addWidget(income_label)

            income_input = QLineEdit()
            income_input.setPlaceholderText("Enter monthly income")
            layout.addWidget(income_input)

            debt_label = QLabel("Debt Ratio")
            layout.addWidget(debt_label)

            debt_input = QLineEdit()
            debt_input.setPlaceholderText("Enter debt ratio (e.g. 0.35)")
            layout.addWidget(debt_input)

            utilization_label = QLabel("Credit Utilization")
            layout.addWidget(utilization_label)

            utilization_input = QLineEdit()
            utilization_input.setPlaceholderText("Enter utilization (e.g. 0.40)")
            layout.addWidget(utilization_input)

            result_title = QLabel("Simulation Result")
            result_title.setObjectName("pageTitle")
            layout.addWidget(result_title)

            result_label = QLabel("Enter borrower details and run simulation.")
            result_label.setObjectName("muted")
            result_label.setWordWrap(True)
            layout.addWidget(result_label)

            simulate_button = QPushButton("RUN RISK SIMULATION")
            layout.addWidget(simulate_button)

            def run_simulation():
                try:
                    income = float(income_input.text())
                    debt = float(debt_input.text())
                    utilization = float(utilization_input.text())

                    if income <= 0 or debt < 0 or utilization < 0:
                        raise ValueError("Enter valid positive values.")

                    project_root = Path(__file__).resolve().parent.parent
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))

                    from modeling.credlens_package.credlens_predictor import CredLensPredictor

                    csv_path = Path(__file__).parent.parent / "cs-training.csv"
                    df = pd.read_csv(csv_path)

                    raw_columns = [
                        "RevolvingUtilizationOfUnsecuredLines",
                        "age",
                        "NumberOfTime30-59DaysPastDueNotWorse",
                        "DebtRatio",
                        "MonthlyIncome",
                        "NumberOfOpenCreditLinesAndLoans",
                        "NumberOfTimes90DaysLate",
                        "NumberRealEstateLoansOrLines",
                        "NumberOfTime60-89DaysPastDueNotWorse",
                        "NumberOfDependents"
                    ]

                    sample = df[raw_columns].median().to_frame().T

                    sample["MonthlyIncome"] = income
                    sample["DebtRatio"] = debt
                    sample["RevolvingUtilizationOfUnsecuredLines"] = utilization

                    predictor = CredLensPredictor()
                    prediction = predictor.predict_df(sample).iloc[0]

                    result_label.setText(
                        f"Prediction: {prediction['label']}\n"
                        f"Risk Category: {prediction['risk_band']}\n"
                        f"Default Probability: {prediction['probability']:.1%}\n"
                        f"Model Version: {prediction['model_version']}\n\n"
                        "Other borrower features use dataset median values."
                    )

                except ValueError as e:
                    result_label.setText(f"Input Error: {e}")

                except Exception as e:
                    result_label.setText(f"Simulation Error: {e}")

            simulate_button.clicked.connect(run_simulation)

        
        elif page_name == "Fairness & Reliability":
            description.setText("Model Reliability & Fairness Overview")
            layout.addSpacing(20)

            try:
                import json

                project_root = Path(__file__).parent.parent

                # Load model configuration
                config_path = (
                    project_root / "modeling" / "credlens_package"
                    / "models" / "model_config.json"
                )

                with open(config_path, "r") as file:
                    config = json.load(file)

                metrics = config["test_metrics_calibrated"]

                title = QLabel("Calibrated Model Test Performance")
                title.setObjectName("pageTitle")
                layout.addWidget(title)

                test_size = config["split"]["n_test"]
                test_note = QLabel(
                    f"Holdout test set: {test_size:,} records | "
                    f"Model Version: {config['model_version']} | "
                    f"Threshold: {metrics['threshold']:.2f}"
                )
                test_note.setObjectName("muted")
                layout.addWidget(test_note)

                metric_info = [
                    ("ROC-AUC", f"{metrics['roc_auc']:.4f}"),
                    ("PR-AUC", f"{metrics['pr_auc']:.4f}"),
                    ("F1 SCORE", f"{metrics['f1']:.4f}"),
                    ("RECALL", f"{metrics['recall']:.4f}"),
                    ("PRECISION", f"{metrics['precision']:.4f}"),
                    ("BALANCED ACCURACY",
                     f"{metrics['balanced_accuracy']:.4f}"),
                    ("BRIER SCORE", f"{metrics['brier']:.4f}"),
                    ("DECISION THRESHOLD",
                     f"{metrics['threshold']:.2f}")
                ]

                metric_grid = QGridLayout()
                metric_grid.setSpacing(16)

                for i, (name, value) in enumerate(metric_info):
                    metric_grid.addWidget(
                        self.create_card(name, value),
                        i // 2, i % 2
                    )

                layout.addLayout(metric_grid)

                # Age-group exploratory analysis
                fairness_title = QLabel("Age Group Fairness Review")
                fairness_title.setObjectName("pageTitle")
                layout.addWidget(fairness_title)

                csv_path = project_root / "cs-training.csv"
                prediction_path = (
                    project_root / "modeling" / "results"
                    / "customer_predictions.csv"
                )

                df = pd.read_csv(csv_path)
                predictions = pd.read_csv(prediction_path)

                if len(df) != len(predictions):
                    raise ValueError(
                        "Dataset and prediction row counts do not match."
                    )

                if "age" not in df.columns:
                    raise ValueError("Age column is missing.")

                if "SeriousDlqin2yrs" not in df.columns:
                    raise ValueError("Target column is missing.")

                review = pd.DataFrame({
                    "age": df["age"],
                    "actual_default": df["SeriousDlqin2yrs"],
                    "probability": predictions["probability"],
                    "risk_band": predictions["risk_band"]
                })

                review["Age Group"] = pd.cut(
                    review["age"],
                    bins=[0, 30, 40, 50, 60, float("inf")],
                    labels=[
                        "Under 30", "30-39", "40-49",
                        "50-59", "60+"
                    ],
                    right=False
                )

                review = review.dropna(
                    subset=["Age Group", "actual_default",
                            "probability", "risk_band"]
                )

                grouped = review.groupby(
                    "Age Group", observed=True
                )

                rows = []

                for age_group, group in grouped:
                    rows.append([
                        str(age_group),
                        f"{len(group):,}",
                        f"{group['actual_default'].mean() * 100:.2f}%",
                        f"{(group['risk_band'] == 'High').mean() * 100:.2f}%",
                        f"{group['probability'].mean() * 100:.2f}%"
                    ])

                table = QTableWidget()
                table.setColumnCount(5)
                table.setHorizontalHeaderLabels([
                    "AGE GROUP",
                    "BORROWERS",
                    "ACTUAL DEFAULT",
                    "HIGH RISK",
                    "AVG. PREDICTED RISK"
                ])
                table.setRowCount(len(rows))

                for row_index, row_data in enumerate(rows):
                    for col_index, value in enumerate(row_data):
                        item = QTableWidgetItem(value)
                        item.setTextAlignment(Qt.AlignCenter)
                        table.setItem(row_index, col_index, item)

                table.setEditTriggers(
                    QAbstractItemView.NoEditTriggers
                )
                table.resizeColumnsToContents()
                table.setMaximumHeight(240)
                layout.addWidget(table)

                fairness_note = QLabel(
                    "This is an exploratory age-group comparison. "
                    "Differences in default rates or predicted risk "
                    "do not independently establish model bias. "
                    "Formal fairness testing requires additional "
                    "group-wise evaluation."
                )
                fairness_note.setObjectName("muted")
                fairness_note.setWordWrap(True)
                layout.addWidget(fairness_note)

                reliability_title = QLabel("Reliability Information")
                reliability_title.setObjectName("pageTitle")
                layout.addWidget(reliability_title)

                reliability_note = QLabel(
                    "Metrics are from the calibrated holdout test "
                    "evaluation supplied with model version "
                    f"{config['model_version']}. "
                    "Age-group results are descriptive, not a "
                    "fairness certification."
                )
                reliability_note.setObjectName("muted")
                reliability_note.setWordWrap(True)
                layout.addWidget(reliability_note)

                layout.addStretch()

            except Exception as e:
                error = QLabel(f"Reliability Error: {e}")
                error.setWordWrap(True)
                layout.addWidget(error)

        elif page_name == "Report Studio":
            description.setText("Credit Risk Project Report Overview")
            layout.addSpacing(20)

            title = QLabel("Project Summary")
            title.setObjectName("pageTitle")
            layout.addWidget(title)

            summary = QLabel(
                "CREDLENS AI is an Explainable Credit Risk Assessment "
                "platform designed to analyze borrower data, understand "
                "default patterns and support risk analysis."
            )
            summary.setWordWrap(True)
            layout.addWidget(summary)

            layout.addSpacing(15)

            sections = [
                ("DATA INTELLIGENCE", "Dataset overview, data quality, missing values and EDA"),
                ("RISK ANALYSIS", "Historical borrower default distribution"),
                ("EXPLAINABILITY", "SHAP-based explanations pending model integration"),
                ("SIMULATION", "Borrower profile input interface ready; prediction pending"),
                ("FAIRNESS", "Group-wise fairness evaluation pending"),
                ("RELIABILITY", "Model validation metrics pending integration")
            ]

            for heading, detail in sections:
                card = self.create_card(heading, detail)
                layout.addWidget(card)

            layout.addSpacing(15)

            note = QLabel(
                "Note: Final model performance and explainability "
                "results will be included after trained model integration."
            )
            note.setObjectName("muted")
            note.setWordWrap(True)
            layout.addWidget(note)

            layout.addStretch()

        else:
            placeholder = QLabel(
                f"{page_name} module will be developed next."
            )
            placeholder.setObjectName("muted")
            layout.addSpacing(25)
            layout.addWidget(placeholder)
            layout.addStretch()

        return page

    def change_page(self, index):
        self.stack.setCurrentIndex(index)

        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CredLensApp()
    window.show()
    sys.exit(app.exec())