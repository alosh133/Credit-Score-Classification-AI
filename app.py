from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Credit Score | Credit Decision Support",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project palette
NAVY = "#10243A"
NAVY_LIGHT = "#183751"
TEAL = "#13A89E"
TEAL_DARK = "#087C78"
BLUE = "#4D8FE8"
PURPLE = "#8974D8"
AMBER = "#E6A83B"
GREEN = "#19866E"
RED = "#D85D67"
INK = "#182B3D"
MUTED = "#6B7C8F"
PALE = "#F3F6FA"
WHITE = "#FFFFFF"

MODEL_NAME = "XGBoost"
PAGE_LABELS = {
    "overview": "🏠  Credit Score Overview",
    "inquiry": "🔎  Inquiry",
    "results": "📊  Results",
}
PAGE_ORDER = list(PAGE_LABELS)

st.markdown(
    f"""<style>
    :root {{
      --cs-navy: {NAVY}; --cs-teal: {TEAL}; --cs-ink: {INK};
      --cs-muted: {MUTED}; --cs-pale: {PALE};
    }}
    .stApp {{ background: var(--cs-pale); color: var(--cs-ink); }}
    [data-testid="stHeader"] {{ background: rgba(243,246,250,.88); }}
    [data-testid="stSidebar"] {{ background: #10243A !important; border-right: 1px solid #203B54; }}
    [data-testid="stSidebar"] * {{ color: #EAF2F8; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: #B8C9D8; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
      background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);
      border-radius: 12px; padding: .6rem .72rem; margin: .16rem 0;
      transition: all .18s ease;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
      background: rgba(19,168,158,.18); border-color: rgba(19,168,158,.48);
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
      background: transparent; border: 0; padding-left: 0; font-weight: 700;
    }}
    /* Keep the collapsed-sidebar control easy to find on the light page. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[title="Expand sidebar"], button[aria-label="Expand sidebar"],
    button[title="Open sidebar"], button[aria-label="Open sidebar"] {{
      background: {NAVY} !important; color: #FFFFFF !important;
      border: 1px solid {TEAL} !important; border-radius: 12px !important;
      min-width: 96px !important; width: 96px !important; min-height: 44px !important;
      opacity: 1 !important; visibility: visible !important;
      box-shadow: 0 5px 14px rgba(16,36,58,.2) !important; z-index: 10000 !important;
    }}
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg,
    button[title="Expand sidebar"] svg, button[aria-label="Expand sidebar"] svg,
    button[title="Open sidebar"] svg, button[aria-label="Open sidebar"] svg {{
      color: #FFFFFF !important; fill: #FFFFFF !important; stroke: #FFFFFF !important;
    }}
    [data-testid="stSidebarCollapsedControl"]::after,
    [data-testid="collapsedControl"]::after,
    button[title="Expand sidebar"]::after, button[aria-label="Expand sidebar"]::after,
    button[title="Open sidebar"]::after, button[aria-label="Open sidebar"]::after {{
      content: " Pages"; color: #FFFFFF; font-size: .8rem; font-weight: 750;
    }}
    .block-container {{ max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem; }}
    h1, h2, h3 {{ color: {NAVY}; letter-spacing: -.025em; }}
    [data-testid="stMetric"] {{
      background: {WHITE}; border: 1px solid #E4EAF0; border-radius: 16px;
      padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(16,36,58,.045);
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 750; }}
    /* The app canvas is light, so keep all main-page form labels and values dark. */
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] label,
    [data-testid="stMain"] label {{ color: {INK} !important; }}
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea {{
      color: {INK} !important; background: {WHITE} !important;
      -webkit-text-fill-color: {INK} !important;
    }}
    [data-testid="stMain"] [data-testid="stNumberInput"] button {{
      color: {NAVY} !important; background: {WHITE} !important;
    }}
    [data-testid="stMain"] [data-baseweb="select"] > div {{
      color: {INK} !important; background: {WHITE} !important; border-color: #D9E3EB !important;
    }}
    [data-testid="stMain"] [data-baseweb="select"] * {{ color: {INK} !important; }}
    [data-testid="stMain"] [data-testid="stRadio"] label {{ color: {INK} !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
      background: {WHITE}; border-color: #E4EAF0 !important; border-radius: 18px;
      box-shadow: 0 5px 20px rgba(16,36,58,.035);
    }}
    .stButton > button {{
      border-radius: 12px; border: 1px solid #D9E3EB; background: {WHITE};
      color: {NAVY}; font-weight: 650; min-height: 2.75rem;
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
    }}
    .stButton > button:hover {{
      border-color: {TEAL}; color: {TEAL_DARK}; transform: translateY(-1px);
      box-shadow: 0 7px 18px rgba(19,168,158,.13);
    }}
    .stButton > button[kind="primary"] {{
      border: 0; color: white; background: linear-gradient(110deg, {TEAL_DARK}, {TEAL});
      box-shadow: 0 7px 18px rgba(19,168,158,.2);
    }}
    .stButton > button[kind="primary"]:hover {{ color: white; filter: brightness(1.04); }}
    [data-testid="stSlider"] [role="slider"] {{ border-color: {TEAL}; }}
    .hero {{
      background: linear-gradient(115deg, {NAVY} 0%, {NAVY_LIGHT} 64%, #176D70 100%);
      color: white; border-radius: 22px; padding: 1.7rem 1.9rem; margin: .35rem 0 1.4rem;
      box-shadow: 0 14px 34px rgba(16,36,58,.13);
    }}
    .hero h1, .hero h2, .hero p {{ color: white !important; margin-bottom: .35rem; }}
    .hero p {{ color: #C9DCE9 !important; }}
    .eyebrow {{ color: #73DDD2; font-size: .76rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }}
    .section-kicker {{ color: {TEAL_DARK}; font-weight: 750; font-size: .76rem; letter-spacing: .1em; text-transform: uppercase; }}
    .chapter-title {{ color: {NAVY}; font-size: 1.18rem; font-weight: 750; margin: .15rem 0 .2rem; }}
    .chapter-note {{ color: {MUTED}; font-size: .88rem; margin: 0 0 .85rem; }}
    .small-note {{ color: {MUTED}; font-size: .82rem; }}
    .tip {{ background: #F0F8F6; border: 1px solid #D9EEE8; border-radius: 13px; padding: .85rem 1rem; margin: .45rem 0; }}
    .tip-title {{ color: {NAVY}; font-weight: 720; margin-bottom: .18rem; }}
    .tip-copy {{ color: #526778; margin: 0; font-size: .91rem; }}
    div[data-testid="stAlert"] {{ border-radius: 14px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    @media (max-width: 760px) {{ .block-container {{ padding-top: 1rem; }} .hero {{ padding: 1.3rem; }} }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    """Load the trained pipeline and its matching label/feature metadata."""
    root = Path(__file__).resolve().parent
    pipeline = joblib.load(root / "credit_score_pipeline.pkl")
    label_encoder = joblib.load(root / "label_encoder.pkl")
    inputs = joblib.load(root / "inputs.pkl")
    return pipeline, label_encoder, list(inputs)


try:
    pipeline, label_encoder, inputs = load_assets()
except FileNotFoundError as exc:
    st.error(f"Required model file not found: {exc.filename}")
    st.info("Place credit_score_pipeline.pkl, label_encoder.pkl, and inputs.pkl beside app.py.")
    st.stop()
except Exception as exc:
    st.error("The credit model could not be loaded.")
    st.exception(exc)
    st.stop()


OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]

# Reference medians retained from the project's exploratory analysis.
# These are project data references, not official FICO thresholds.
GOOD_MEDIAN = {"debt": 737, "rate": 7, "delay": 10, "loans": 2, "inq": 3, "cards": 4}
GOOD_PROFILE = {
    "debt": 740.0, "rate": 7.0, "delay": 10.0,
    "loans": 2.0, "inq": 3.0, "cards": 4.0,
}
REFERENCE_LABELS = {
    "debt": "Outstanding debt ($)",
    "rate": "Interest rate (%)",
    "delay": "Days past due",
    "loans": "Number of loans",
    "inq": "Credit inquiries",
    "cards": "Credit cards",
}

BASE = dict(
    occ="Engineer", sal=3100.0, bal=340.0, inv=120.0, bank=5, debt=1000.0,
    emi=63.0, cards=5, loans=3, rate=13, mix="Standard", hist=18, delay=18,
    util=32.0, chg=10.0, inq=5, late=14, minpay="Yes",
)
PROFILES = {
    "🟢 Good-like example": {
        **BASE, "sal": 3800.0, "bal": 400.0, "inv": 150.0, "bank": 3,
        "debt": 740.0, "emi": 58.0, "cards": 4, "loans": 2, "rate": 7,
        "mix": "Good", "hist": 24, "delay": 10, "util": 31.0, "chg": 8.0,
        "inq": 3, "late": 9, "minpay": "No",
    },
    "🟡 Standard-like example": BASE,
    "🔴 Poor-like example": {
        **BASE, "sal": 2700.0, "bal": 280.0, "inv": 80.0, "bank": 7,
        "debt": 1950.0, "emi": 75.0, "cards": 7, "loans": 5, "rate": 21,
        "mix": "Bad", "hist": 12, "delay": 27, "util": 33.0, "inq": 8,
        "late": 17,
    },
}

for key, value in BASE.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("page", "overview")


def go_to(page):
    st.session_state.page = page


def load_profile(name):
    st.session_state.update(PROFILES[name])


def section_header(label, title, description):
    st.markdown(
        f'<div class="section-kicker">{label}</div>'
        f'<div class="chapter-title">{title}</div>'
        f'<div class="chapter-note">{description}</div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=330, **extra):
    layout = dict(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=INK, size=12),
        margin=dict(t=28, b=28, l=18, r=18),
        hoverlabel=dict(bgcolor=WHITE, font_color=INK),
    )
    layout.update(extra)
    fig.update_layout(**layout)
    return fig


def build_model_row(s):
    delay_risk = s.late * (max(s.delay // 30, 0) + 1) + (5 if s.minpay == "Yes" else 0)
    # Retained exactly as the project's trained feature definition.
    project_debt_income = s.debt / (s.sal * 12) * 100
    row = {
        "Occupation": s.occ,
        "Monthly_Inhand_Salary": s.sal,
        "Num_Bank_Accounts": s.bank,
        "Num_Credit_Card": s.cards,
        "Interest_Rate": s.rate,
        "Num_of_Loan": s.loans,
        "Delay_from_due_date": s.delay,
        "Changed_Credit_Limit": s.chg,
        "Num_Credit_Inquiries": s.inq,
        "Credit_Mix": s.mix,
        "Outstanding_Debt": s.debt,
        "Credit_Utilization_Ratio": s.util,
        "Credit_History_Age": s.hist,
        "Total_EMI_per_month": s.emi,
        "Amount_invested_monthly": s.inv,
        "Monthly_Balance": s.bal,
        "Payment_History_Risk": delay_risk,
        "Debt_to_Income_Ratio": project_debt_income,
    }
    return row, delay_risk, project_debt_income


def predict_customer(row):
    missing = [column for column in inputs if column not in row]
    if missing:
        raise ValueError(f"The model expects input columns that this app does not provide: {missing}")
    frame = pd.DataFrame([row], columns=inputs)
    encoded_prediction = np.asarray(pipeline.predict(frame)).ravel()
    predicted_label = label_encoder.inverse_transform(encoded_prediction.astype(int))[0]

    # Match probability columns to the estimator's actual encoded class order.
    encoded_classes = getattr(pipeline, "classes_", np.arange(len(label_encoder.classes_)))
    probabilities = pipeline.predict_proba(frame)[0]
    probability_by_label = {
        label_encoder.inverse_transform(np.asarray([int(class_id)]))[0]: float(probability)
        for class_id, probability in zip(encoded_classes, probabilities)
    }
    return predicted_label, probability_by_label


def hero(kicker, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# Sidebar navigation
with st.sidebar:
    st.markdown(
        f"""<div style="padding:.5rem .25rem 1.2rem">
        <div style="font-size:2rem">💳</div>
        <div style="color:#fff;font-size:1.22rem;font-weight:800;letter-spacing:.02em">CREDIT SCORE</div>
        <div style="color:#91A8BB;font-size:.78rem;margin-top:.2rem">Credit decision support</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='color:#91A8BB;font-size:.7rem;font-weight:700;letter-spacing:.12em;margin:.25rem 0 .5rem'>WORKSPACE</div>", unsafe_allow_html=True)
    st.radio(
        "Navigate",
        PAGE_ORDER,
        key="page",
        format_func=lambda page: PAGE_LABELS[page],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#284156;margin:1.3rem 0'>", unsafe_allow_html=True)
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Educational decision-support prototype")

s = st.session_state
st.caption("Navigation is in the left sidebar. If it is hidden, click the » arrow in the upper-left corner to reopen it.")


# ── Page 1: Credit Score Overview ───────────────────────────────────────────
if s.page == "overview":
    hero(
        "CREDIT DECISION SUPPORT · PROJECT OVERVIEW",
        "Understand the Credit Score",
        "A clear introduction to the scoring language, the factors behind it, and what this project predicts.",
    )

    intro, purpose = st.columns([1.12, 0.88], gap="large")
    with intro:
        with st.container(border=True):
            st.markdown('<div class="section-kicker">The scoring language</div>', unsafe_allow_html=True)
            st.subheader("Why do we use the term FICO?")
            st.write(
                "FICO is a widely recognized credit-scoring framework created by Fair Isaac Corporation. "
                "We use its name here as familiar terminology for discussing credit classification and the "
                "types of information commonly associated with credit scoring."
            )
            st.write(
                "A FICO Score is one kind of credit score. This project does not calculate an official FICO Score: "
                "it uses a separate machine-learning model to classify the supplied profile as Good, Standard, or Poor."
            )
            st.markdown(
                "<div class='small-note'>The FICO factor chart is an educational reference. Actual scoring models and factor impact can vary.</div>",
                unsafe_allow_html=True,
            )
    with purpose:
        with st.container(border=True):
            st.markdown('<div class="section-kicker">Why credit scoring matters</div>', unsafe_allow_html=True)
            st.subheader("A consistent view of credit risk")
            st.write(
                "Lenders may use credit scores alongside an application and other information to help assess risk. "
                "A score can support consistent review, but it does not guarantee approval, a particular rate, or a credit limit."
            )
            st.write(
                f"This project evaluates {len(inputs)} model features and provides a predicted class, probability estimates, "
                "and educational observations for review."
            )

    st.markdown("### The five commonly cited FICO factor categories")
    chart_col, details_col = st.columns([1.05, 1], gap="large")
    factor_names = ["Payment history", "Amounts owed", "Credit history length", "New credit", "Credit mix"]
    factor_values = [35, 30, 15, 10, 10]
    factor_colors = [NAVY, BLUE, AMBER, TEAL, PURPLE]
    with chart_col:
        with st.container(border=True):
            factor_fig = go.Figure(go.Pie(
                labels=factor_names, values=factor_values, hole=0.66, sort=False,
                marker=dict(colors=factor_colors, line=dict(color=WHITE, width=3)),
                textinfo="percent", textposition="inside", insidetextfont=dict(color=WHITE, size=13),
                hovertemplate="%{label}<br>Reference share: %{value}%<extra></extra>",
            ))
            factor_fig.add_annotation(text="FICO<br><b>FACTORS</b>", x=0.5, y=0.5, showarrow=False,
                                      font=dict(size=15, color=NAVY))
            chart_layout(factor_fig, height=390, showlegend=True,
                         legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                         margin=dict(t=12, b=38, l=8, r=8))
            st.plotly_chart(factor_fig, width="stretch", theme=None, config={"displayModeBar": False})
    with details_col:
        descriptions = [
            ("Payment history", "35%", "Whether accounts have generally been paid on time."),
            ("Amounts owed", "30%", "Balances and the amount of available revolving credit being used."),
            ("Credit history length", "15%", "How long credit accounts have been established."),
            ("New credit", "10%", "Recent credit activity and applications."),
            ("Credit mix", "10%", "The types of credit accounts shown in the report."),
        ]
        for index, (name, weight, description) in enumerate(descriptions):
            with st.container(border=True):
                left, right = st.columns([1, 0.22])
                left.markdown(f"**{name}**  \n<span class='small-note'>{description}</span>", unsafe_allow_html=True)
                right.markdown(f"<div style='text-align:right;color:{factor_colors[index]};font-size:1.1rem;font-weight:800'>{weight}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='small-note'>These published category weights describe a common FICO reference, not this project's model feature importance.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "Source: <a href='https://www.myfico.com/credit-education/whats-in-your-credit-score' target='_blank'>myFICO — What's in your credit score?</a>",
        unsafe_allow_html=True,
    )
    st.button("Continue to Inquiry  →", type="primary", on_click=go_to, args=("inquiry",))


# ── Page 2: Inquiry ─────────────────────────────────────────────────────────
elif s.page == "inquiry":
    hero(
        "CUSTOMER PROFILE · MODEL INPUTS",
        "Inquiry",
        "Enter the customer's profile. Inputs are grouped under the five familiar FICO factors, followed by the model's additional inputs.",
    )
    st.info(
        "This classifier is a project model, not an official FICO calculator. Some model inputs are not FICO factors; "
        "they are shown in a separate section so the model receives its full feature set."
    )

    st.markdown("#### Quick profile examples")
    st.caption("Examples prefill the form for demonstration. They do not guarantee a specific model result.")
    profile_cols = st.columns(3, gap="medium")
    for column, profile_name in zip(profile_cols, PROFILES):
        column.button(profile_name, on_click=load_profile, args=(profile_name,), width="stretch")

    st.markdown("### Customer information")
    pay_col, owed_col = st.columns(2, gap="large")
    with pay_col:
        with st.container(border=True):
            section_header("FACTOR 01", "Payment History", "Repayment behavior and past-due activity.")
            st.slider("Days past due", -5, 67, key="delay", help="Project input for payment-delay history.")
            st.slider("Number of delayed payments", 0, 28, key="late")
            st.radio("Pays only the minimum due?", ["Yes", "No"], key="minpay", horizontal=True)
    with owed_col:
        with st.container(border=True):
            section_header("FACTOR 02", "Amounts Owed", "Current debt, revolving-credit use, and monthly obligations.")
            st.number_input("Outstanding debt ($)", 0.0, 5000.0, step=50.0, key="debt")
            st.slider("Credit utilization (%)", 20.0, 50.0, key="util", help="Value range follows the training data used by this project.")
            st.number_input("Monthly EMI ($)", 0.0, 1800.0, step=10.0, key="emi")

    history_col, new_col = st.columns(2, gap="large")
    with history_col:
        with st.container(border=True):
            section_header("FACTOR 03", "Credit History Length", "How long the customer's credit history has been established.")
            st.slider("Credit history age (years)", 0, 33, key="hist")
    with new_col:
        with st.container(border=True):
            section_header("FACTOR 04", "New Credit", "Recent inquiries and credit activity. These are grouped for this project; the model is not an official FICO scorecard.")
            st.slider("Number of credit inquiries", 0, 17, key="inq")
            st.slider("Changed credit limit (%)", -7.0, 37.0, key="chg")
            st.slider("Interest rate (%)", 1, 34, key="rate")

    mix_col, financial_col = st.columns(2, gap="large")
    with mix_col:
        with st.container(border=True):
            section_header("FACTOR 05", "Credit Mix", "Credit types and accounts represented in the model inputs.")
            st.selectbox("Credit mix category", ["Bad", "Standard", "Good"], key="mix")
            mix_a, mix_b = st.columns(2)
            mix_a.slider("Number of credit cards", 0, 10, key="cards")
            mix_b.slider("Number of loans", 0, 9, key="loans")
            st.slider("Number of bank accounts", 0, 10, key="bank")
    with financial_col:
        with st.container(border=True):
            section_header("ADDITIONAL INPUTS", "Additional Model Inputs", "Financial and personal variables used by this project model; these are not official FICO factor categories.")
            st.selectbox("Occupation", OCCUPATIONS, key="occ")
            st.number_input("Monthly take-home income ($)", 300.0, 15500.0, step=100.0, key="sal")
            extra_a, extra_b = st.columns(2)
            extra_a.number_input("Monthly balance ($)", 0.0, 1700.0, step=25.0, key="bal")
            extra_b.number_input("Monthly amount invested ($)", 0.0, 2000.0, step=25.0, key="inv")

    back_col, next_col = st.columns([1, 1])
    back_col.button("← Back to Overview", on_click=go_to, args=("overview",), width="stretch")
    next_col.button("Generate Customer Results  →", type="primary", on_click=go_to, args=("results",), width="stretch")


# ── Page 3: Results ──────────────────────────────────────────────────────────
else:
    hero(
        "CUSTOMER ANALYSIS · MODEL OUTPUT",
        "Results",
        "Review the predicted class, three customer charts, and practical observations based on the entered profile.",
    )
    edit_col, overview_col = st.columns([1, 1])
    edit_col.button("← Edit Customer Inquiry", on_click=go_to, args=("inquiry",), width="stretch")
    overview_col.button("🏠 Back to Credit Score Overview", on_click=go_to, args=("overview",), width="stretch")
    row, payment_risk, project_dti = build_model_row(s)
    try:
        label, probabilities = predict_customer(row)
    except Exception as exc:
        st.error("The model could not score this customer profile.")
        st.exception(exc)
        st.stop()

    class_colors = {"Good": GREEN, "Standard": AMBER, "Poor": RED}
    label_color = class_colors.get(str(label), BLUE)
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("Predicted class", str(label))
    m2.metric("Model confidence", f"{probabilities.get(label, 0.0) * 100:.1f}%")
    m3.metric("Project debt/income indicator", f"{project_dti:.1f}%")
    m4.metric("Payment-history risk input", f"{payment_risk:.0f}")
    st.markdown(
        "<div class='small-note'>Confidence is the model's class probability estimate. The debt/income indicator follows the project's trained feature formula and is not a standard lender DTI calculation.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Customer Snapshot")
    chart_cols = st.columns(3, gap="large")
    with chart_cols[0]:
        with st.container(border=True):
            st.markdown("**Predicted class probabilities**")
            order = [name for name in ["Poor", "Standard", "Good"] if name in probabilities]
            prob_fig = go.Figure(go.Bar(
                x=[probabilities[name] * 100 for name in order], y=order, orientation="h",
                marker_color=[class_colors.get(name, BLUE) for name in order],
                text=[f"{probabilities[name] * 100:.1f}%" for name in order], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            chart_layout(prob_fig, height=310, xaxis=dict(range=[0, 115], title="Probability (%)", gridcolor="#E8EDF2"),
                         yaxis=dict(title="", autorange="reversed"), showlegend=False,
                         margin=dict(t=24, b=36, l=8, r=34))
            st.plotly_chart(prob_fig, width="stretch", theme=None, config={"displayModeBar": False})

    with chart_cols[1]:
        with st.container(border=True):
            st.markdown("**Selected inputs vs Good-like reference**")
            keys = list(GOOD_PROFILE)
            ratios = [min(float(s[key]) / GOOD_PROFILE[key] * 100, 400) for key in keys]
            ratio_fig = go.Figure(go.Bar(
                x=ratios, y=[REFERENCE_LABELS[key] for key in keys], orientation="h",
                marker_color=[label_color if value <= 100 else AMBER for value in ratios],
                text=[f"{value:.0f}%" for value in ratios], textposition="outside",
                hovertemplate="%{y}<br>Customer / reference: %{x:.1f}%<extra></extra>",
            ))
            ratio_fig.add_vline(x=100, line_dash="dash", line_color=MUTED, annotation_text="Reference")
            chart_layout(ratio_fig, height=310, xaxis=dict(range=[0, 460], title="% of reference value", gridcolor="#E8EDF2"),
                         yaxis=dict(title="", autorange="reversed"), showlegend=False,
                         margin=dict(t=24, b=36, l=8, r=34))
            st.plotly_chart(ratio_fig, width="stretch", theme=None, config={"displayModeBar": False})

    with chart_cols[2]:
        with st.container(border=True):
            st.markdown("**Monthly financial snapshot**")
            financial_names = ["Take-home income", "Monthly EMI", "Invested", "Balance"]
            financial_values = [s.sal, s.emi, s.inv, s.bal]
            financial_colors = [NAVY, RED, BLUE, TEAL]
            money_fig = go.Figure(go.Bar(
                x=financial_names, y=financial_values,
                marker_color=financial_colors,
                text=[f"${value:,.0f}" for value in financial_values], textposition="outside",
                hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
            ))
            chart_layout(money_fig, height=310, yaxis=dict(title="Amount ($)", gridcolor="#E8EDF2", rangemode="tozero"),
                         showlegend=False, margin=dict(t=24, b=70, l=8, r=8))
            st.plotly_chart(money_fig, width="stretch", theme=None, config={"displayModeBar": False})

    st.markdown("### Customer Guidance")
    tips = []
    if s.delay > GOOD_MEDIAN["delay"] or s.late > 9:
        tips.append(("Payment history", "Review past-due accounts and aim to make future payments by their due dates."))
    if s.minpay == "Yes":
        tips.append(("Monthly payments", "If your budget allows, paying more than the minimum may help reduce the balance faster. Keep payments affordable."))
    if s.debt > GOOD_MEDIAN["debt"] * 1.5:
        tips.append(("Outstanding debt", "Review your current balances and repayment plan. The entered debt is above the project's Good-segment reference."))
    if s.util > 30:
        tips.append(("Credit utilization", "Review how much of your available revolving credit you use. The 30% comparison shown here is a project guideline, not a universal FICO cutoff."))
    if s.loans > GOOD_MEDIAN["loans"]:
        tips.append(("Credit commitments", "Consider your existing repayment commitments before taking on additional borrowing."))
    if s.inq > GOOD_MEDIAN["inq"]:
        tips.append(("Recent applications", "Before applying for new credit, compare options and apply when you have a clear need."))
    if s.mix == "Bad":
        tips.append(("Credit mix", "Check that the credit-mix information is accurate. Do not open an account solely to change this model's result."))
    if not tips:
        tips.append(("Keep your information current", "Your entered indicators are near the project's reference profile. Continue reviewing account information for accuracy."))

    for title, copy in tips:
        st.markdown(
            f"<div class='tip'><div class='tip-title'>✓ &nbsp;{title}</div><p class='tip-copy'>{copy}</p></div>",
            unsafe_allow_html=True,
        )

    st.caption(
        "Guidance is educational and based on this project's inputs and reference values. "
        "It is not a lender decision or a promise that a credit score will change. Verify credit-report information with its provider."
    )

    with st.expander("How to read these results"):
        st.write(
            "The classifier is trained for Good, Standard, and Poor categories. Its probabilities describe the model's estimates, "
            "not the probability of loan approval. The reference chart compares selected values with medians from the project's "
            "Good-segment exploratory analysis; it is not a FICO scorecard or a causal explanation of the model."
        )
