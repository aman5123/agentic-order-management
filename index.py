"""
Agentic AI Order Management System — Streamlit Demo
A multi-agent order processing pipeline where specialized AI agents
(Intake, Inventory, Fraud Detection, Payment, Logistics, Notification)
collaborate under an Orchestrator to process customer orders end to end.

Run with:
    streamlit run agentic_order_management.py
"""

import time
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
st.set_page_config(
    page_title="Agentic AI Order Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- Header ---------- */
    .main-header {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(79,70,229,0.25);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
        font-weight: 800;
    }
    .main-header p {
        color: #f1edff !important;
        font-size: 1.02rem;
        margin: 0;
    }

    /* ---------- Metric cards (explicit dark text so they never inherit the app's white theme text) ---------- */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        border: 1px solid #ececec;
    }
    .metric-card .metric-label {
        color: #6b7280 !important;
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card .metric-value {
        color: #111827 !important;
        margin: 0.15rem 0 0 0;
        font-size: 1.9rem;
        font-weight: 800;
    }

    /* ---------- Agent cards ---------- */
    .agent-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.16);
        border-left: 5px solid #7c3aed;
        margin-bottom: 0.9rem;
        transition: transform 0.15s ease;
    }
    .agent-card:hover { transform: translateY(-3px); }
    .agent-card h4 { margin-bottom: 0.25rem; color: #111827 !important; }
    .agent-card p { color: #4b5563 !important; font-size: 0.9rem; margin-bottom: 0; }
    .agent-card .agent-name { color: #111827 !important; font-size: 0.85rem; font-weight: 700; }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-success { background:#dcfce7; color:#15803d !important; }
    .badge-warn    { background:#fef3c7; color:#b45309 !important; }
    .badge-fail    { background:#fee2e2; color:#b91c1c !important; }
    .badge-info    { background:#dbeafe; color:#1d4ed8 !important; }

    /* ---------- Live agent log lines ---------- */
    .log-line {
        font-family: 'Consolas', monospace;
        font-size: 0.86rem;
        padding: 0.5rem 0.7rem;
        border-radius: 8px;
        margin-bottom: 0.35rem;
        background: #ffffff;
        color: #1f2937 !important;
        border-left: 3px solid #7c3aed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }

    .section-title {
        font-weight: 800;
        font-size: 1.3rem;
        margin: 1.1rem 0 0.6rem 0;
        color: inherit;
    }

    /* ---------- Feature / home cards ---------- */
    .feature-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.16);
        height: 100%;
        border-top: 4px solid #7c3aed;
    }
    .feature-card h3 { color: #111827 !important; margin: 0.3rem 0 0.4rem 0; font-size: 1.05rem; }
    .feature-card p  { color: #4b5563 !important; font-size: 0.88rem; margin: 0; }
    .feature-card .feature-icon { font-size: 1.8rem; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 55%, #4c1d95 100%);
    }
    section[data-testid="stSidebar"] * { color: #ede9fe !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0;
    }
    .sidebar-caption {
        font-size: 0.8rem;
        color: #c4b5fd !important;
        margin-bottom: 1rem;
    }

    /* Nav buttons in the sidebar */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #ede9fe !important;
        text-align: left;
        border-radius: 10px;
        padding: 0.55rem 0.9rem;
        font-weight: 600;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: rgba(255,255,255,0.16);
        border-color: rgba(255,255,255,0.3);
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(120deg, #7c3aed, #db2777);
        border: none;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(124,58,237,0.45);
    }

    .sidebar-stat-box {
        background: rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-top: 0.6rem;
        font-size: 0.82rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# STATIC DATA
# ----------------------------------------------------------------------------
PRODUCTS = {
    "Wireless Headphones": 79.99,
    "Smart Watch": 149.99,
    "Mechanical Keyboard": 109.50,
    "4K Monitor": 299.00,
    "Gaming Mouse": 49.99,
    "USB-C Hub": 34.99,
    "Bluetooth Speaker": 59.99,
    "Laptop Stand": 29.99,
}

COURIERS = ["BlueDart Express", "FedEx Priority", "DHL Swift", "Local Same-Day"]

AGENTS = [
    {"key": "orchestrator", "icon": "🧭", "name": "Orchestrator Agent",
     "role": "Coordinates the whole pipeline, routes the order between specialist agents, and makes the final go/no-go call."},
    {"key": "intake", "icon": "📝", "name": "Intake Agent",
     "role": "Validates order data — customer details, product, quantity and address — before anything else runs."},
    {"key": "inventory", "icon": "📦", "name": "Inventory Agent",
     "role": "Checks real-time stock levels and reserves units for the order, or flags a backorder."},
    {"key": "fraud", "icon": "🛡️", "name": "Fraud Detection Agent",
     "role": "Scores the order for risk using amount, velocity and behavioral signals; escalates suspicious orders."},
    {"key": "payment", "icon": "💳", "name": "Payment Agent",
     "role": "Authorizes and captures payment through the selected method, retries on soft declines."},
    {"key": "logistics", "icon": "🚚", "name": "Logistics Agent",
     "role": "Selects the optimal courier and warehouse, and calculates an estimated delivery window."},
    {"key": "notify", "icon": "📧", "name": "Notification Agent",
     "role": "Sends the customer confirmation, tracking info, and keeps them updated at each milestone."},
]

STATUS_COLORS = {
    "Completed": "badge-success",
    "Flagged for Review": "badge-warn",
    "Failed": "badge-fail",
    "Processing": "badge-info",
}

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "orders" not in st.session_state:
    st.session_state.orders = []
if "order_counter" not in st.session_state:
    st.session_state.order_counter = 1000

# ----------------------------------------------------------------------------
# AGENT LOGIC (simulated "reasoning")
# ----------------------------------------------------------------------------
def run_intake_agent(order):
    ok = order["quantity"] > 0 and order["customer"].strip() != ""
    msg = (f"Validated customer '{order['customer']}', product '{order['product']}', "
           f"qty {order['quantity']}. Data schema OK.") if ok else "Missing/invalid required fields."
    return ok, msg


def run_inventory_agent(order):
    stock = random.randint(0, 40)
    available = stock >= order["quantity"]
    if available:
        msg = f"Stock check passed — {stock} units available, {order['quantity']} reserved from Warehouse-{random.randint(1,4)}."
    else:
        msg = f"Insufficient stock — only {stock} units available for requested {order['quantity']}. Backorder created."
    return available, msg


def run_fraud_agent(order):
    base_risk = min(order["amount"] / 20, 40)
    noise = random.uniform(0, 35)
    risk_score = round(base_risk + noise, 1)
    flagged = risk_score > 55
    msg = (f"Computed risk score {risk_score}/100 (amount, device & velocity signals). "
           + ("⚠️ Escalated for manual review." if flagged else "Within safe threshold, cleared."))
    return risk_score, flagged, msg


def run_payment_agent(order, flagged):
    if flagged:
        return False, "Payment authorization held pending fraud review."
    success = random.random() > 0.06
    if success:
        txn = uuid.uuid4().hex[:10].upper()
        msg = f"Payment of ${order['amount']:.2f} via {order['payment_method']} authorized & captured. Txn #{txn}"
    else:
        msg = f"Payment via {order['payment_method']} declined by issuer. Order halted."
    return success, msg


def run_logistics_agent(order):
    courier = random.choice(COURIERS)
    days = random.randint(1, 6)
    eta = (datetime.now() + timedelta(days=days)).strftime("%d %b %Y")
    msg = f"Assigned courier '{courier}'. Estimated delivery by {eta} ({days} day(s))."
    return courier, eta, msg


def run_notification_agent(order, status):
    if status == "Completed":
        msg = f"Confirmation + tracking link emailed to {order['customer']}."
    elif status == "Flagged for Review":
        msg = f"Notified {order['customer']} that their order is under review."
    else:
        msg = f"Notified {order['customer']} that the order could not be processed."
    return msg


def process_order_pipeline(order, live_container):
    """Runs every agent in sequence, streaming live status into live_container."""
    log = []
    status_box = live_container.container()

    def emit(icon, name, text, kind="info"):
        badge = {"info": "badge-info", "success": "badge-success",
                 "warn": "badge-warn", "fail": "badge-fail"}[kind]
        ts = datetime.now().strftime("%H:%M:%S")
        status_box.markdown(
            f"<div class='log-line'>{icon} <b>{name}</b> "
            f"<span class='badge {badge}'>{kind.upper()}</span> "
            f"&nbsp;<span style='color:#888'>{ts}</span><br>{text}</div>",
            unsafe_allow_html=True,
        )
        log.append({"time": ts, "agent": name, "message": text, "kind": kind})

    emit("🧭", "Orchestrator", f"New order received from {order['customer']} — dispatching to agent network.", "info")
    time.sleep(0.35)

    ok, msg = run_intake_agent(order)
    emit("📝", "Intake Agent", msg, "success" if ok else "fail")
    time.sleep(0.35)
    if not ok:
        emit("🧭", "Orchestrator", "Pipeline halted — intake validation failed.", "fail")
        return "Failed", log, {}

    avail, msg = run_inventory_agent(order)
    emit("📦", "Inventory Agent", msg, "success" if avail else "fail")
    time.sleep(0.35)
    if not avail:
        emit("🧭", "Orchestrator", "Pipeline halted — item(s) not available in inventory.", "fail")
        run_notification_agent(order, "Failed")
        return "Failed", log, {}

    risk_score, flagged, msg = run_fraud_agent(order)
    emit("🛡️", "Fraud Detection Agent", msg, "warn" if flagged else "success")
    time.sleep(0.35)

    paid, msg = run_payment_agent(order, flagged)
    emit("💳", "Payment Agent", msg, "success" if paid else ("warn" if flagged else "fail"))
    time.sleep(0.35)

    extra = {"risk_score": risk_score}

    if flagged:
        emit("🧭", "Orchestrator", "Order routed to human-in-the-loop review queue due to elevated fraud risk.", "warn")
        run_notification_agent(order, "Flagged for Review")
        emit("📧", "Notification Agent", run_notification_agent(order, "Flagged for Review"), "warn")
        return "Flagged for Review", log, extra

    if not paid:
        emit("🧭", "Orchestrator", "Pipeline halted — payment could not be captured.", "fail")
        emit("📧", "Notification Agent", run_notification_agent(order, "Failed"), "fail")
        return "Failed", log, extra

    courier, eta, msg = run_logistics_agent(order)
    emit("🚚", "Logistics Agent", msg, "success")
    time.sleep(0.35)
    extra["courier"] = courier
    extra["eta"] = eta

    notif_msg = run_notification_agent(order, "Completed")
    emit("📧", "Notification Agent", notif_msg, "success")
    time.sleep(0.2)

    emit("🧭", "Orchestrator", "All agents completed successfully. Order finalized. ✅", "success")
    return "Completed", log, extra


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
NAV_ITEMS = [
    "🏡 Home",
    "🏠 Dashboard",
    "🛒 New Order",
    "📋 Order Tracking",
    "🕸️ Agent Network",
    "📊 Analytics",
]

if "page" not in st.session_state:
    st.session_state.page = "🏡 Home"

with st.sidebar:
    st.markdown("<p class='sidebar-brand'>🤖 Agentic OMS</p>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-caption'>Multi-agent order processing demo</p>", unsafe_allow_html=True)

    for item in NAV_ITEMS:
        is_active = st.session_state.page == item
        if st.button(item, key=f"nav_{item}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = item
            st.rerun()

    st.markdown("---")

    orders = st.session_state.orders
    n_total = len(orders)
    n_completed = sum(1 for o in orders if o["status"] == "Completed")
    st.markdown(
        f"<div class='sidebar-stat-box'>📦 Orders this session: <b>{n_total}</b><br>"
        f"✅ Completed: <b>{n_completed}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset demo data", use_container_width=True):
        st.session_state.orders = []
        st.session_state.order_counter = 1000
        st.rerun()

page = st.session_state.page

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Agentic AI Order Management System</h1>
        <p>Autonomous agents collaborate — intake, inventory, fraud, payment, logistics & notification — to process every order end-to-end.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# PAGE: HOME
# ----------------------------------------------------------------------------
if page == "🏡 Home":
    orders = st.session_state.orders
    total = len(orders)
    completed = sum(1 for o in orders if o["status"] == "Completed")
    revenue = sum(o["amount"] for o in orders if o["status"] == "Completed")

    c1, c2, c3 = st.columns(3)
    for col, label, value in zip([c1, c2, c3],
                                  ["Orders Processed", "Completed", "Revenue Generated"],
                                  [total, completed, f"${revenue:,.2f}"]):
        with col:
            st.markdown(
                f"<div class='metric-card'><p class='metric-label'>{label}</p>"
                f"<p class='metric-value'>{value}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>What is Agentic OMS?</div>", unsafe_allow_html=True)
    st.write(
        "A demonstration of **agentic AI applied to order management**. Instead of one monolithic "
        "system, seven specialized agents each own a single responsibility — validating orders, "
        "checking stock, scoring fraud risk, capturing payment, arranging logistics, and notifying "
        "customers — all coordinated by an Orchestrator agent. Every decision is logged in plain "
        "language so you can see exactly how the AI reasoned at each step."
    )

    st.markdown("<div class='section-title'>Explore the Demo</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    features = [
        (f1, "🛒", "Place an Order", "Submit a new order and watch the agent pipeline process it live, step by step, in real time."),
        (f2, "📋", "Track Orders", "Browse every order processed this session and drill into the full agent-by-agent decision log."),
        (f3, "🕸️", "Agent Network", "See how the agents hand off work to one another in a visual orchestration diagram."),
    ]
    for col, icon, title, desc in features:
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon'>{icon}</div>"
                f"<h3>{title}</h3><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    cta_col = st.columns([1, 1, 1])[1]
    with cta_col:
        if st.button("🚀 Create Your First Order", use_container_width=True, type="primary"):
            st.session_state.page = "🛒 New Order"
            st.rerun()

# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------
elif page == "🏠 Dashboard":
    orders = st.session_state.orders

    total = len(orders)
    completed = sum(1 for o in orders if o["status"] == "Completed")
    flagged = sum(1 for o in orders if o["status"] == "Flagged for Review")
    failed = sum(1 for o in orders if o["status"] == "Failed")
    revenue = sum(o["amount"] for o in orders if o["status"] == "Completed")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, delta in zip(
        [c1, c2, c3, c4, c5],
        ["Total Orders", "Completed", "Flagged", "Failed", "Revenue"],
        [total, completed, flagged, failed, f"${revenue:,.2f}"],
        ["", "✅", "⚠️", "❌", "💰"],
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><p class='metric-label'>{delta} {label}</p>"
                f"<p class='metric-value'>{value}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Live Agent Network Status</div>", unsafe_allow_html=True)
    cols = st.columns(len(AGENTS))
    for col, agent in zip(cols, AGENTS):
        with col:
            st.markdown(
                f"<div class='agent-card' style='text-align:center;'>"
                f"<div style='font-size:1.8rem'>{agent['icon']}</div>"
                f"<span class='agent-name'>{agent['name'].replace(' Agent','')}</span><br>"
                f"<span class='badge badge-success'>● Online</span></div>",
                unsafe_allow_html=True,
            )

    if total > 0:
        left, right = st.columns([1, 1.2])
        with left:
            df_status = pd.DataFrame(
                {"Status": ["Completed", "Flagged for Review", "Failed"],
                 "Count": [completed, flagged, failed]}
            )
            fig = px.pie(df_status, names="Status", values="Count", hole=0.55,
                         color="Status",
                         color_discrete_map={"Completed": "#22c55e", "Flagged for Review": "#f59e0b", "Failed": "#ef4444"})
            fig.update_layout(title="Order Outcomes", margin=dict(t=40, b=0, l=0, r=0), height=340)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            df = pd.DataFrame(orders)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df_time = df.groupby(df["timestamp"].dt.floor("min")).size().reset_index(name="orders")
            fig2 = px.bar(df_time, x="timestamp", y="orders", title="Orders Over Time")
            fig2.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=340)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No orders processed yet — head to **🛒 New Order** to trigger the agent pipeline.")

# ----------------------------------------------------------------------------
# PAGE: NEW ORDER
# ----------------------------------------------------------------------------
elif page == "🛒 New Order":
    st.markdown("<div class='section-title'>Create a New Order</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.3])

    with left:
        with st.form("new_order_form"):
            customer = st.text_input("Customer Name", placeholder="e.g. Aditi Sharma")
            product = st.selectbox("Product", list(PRODUCTS.keys()))
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            payment_method = st.selectbox("Payment Method", ["Credit Card", "UPI", "PayPal", "Net Banking"])
            address = st.text_area("Delivery Address", placeholder="123 MG Road, Karnal, Haryana")
            submitted = st.form_submit_button("🚀 Submit Order to Agents", use_container_width=True)

        st.markdown("<div class='section-title'>Agents in this Pipeline</div>", unsafe_allow_html=True)
        for agent in AGENTS:
            st.markdown(
                f"<div class='agent-card'><h4>{agent['icon']} {agent['name']}</h4>"
                f"<p>{agent['role']}</p></div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("<div class='section-title'>Live Agent Execution</div>", unsafe_allow_html=True)
        live_area = st.empty()

        if submitted:
            if not customer.strip():
                st.error("Please enter a customer name.")
            else:
                unit_price = PRODUCTS[product]
                amount = round(unit_price * quantity, 2)
                order_id = f"ORD-{st.session_state.order_counter}"
                st.session_state.order_counter += 1

                order = {
                    "order_id": order_id,
                    "customer": customer,
                    "product": product,
                    "quantity": quantity,
                    "amount": amount,
                    "payment_method": payment_method,
                    "address": address,
                    "timestamp": datetime.now(),
                }

                with st.spinner("Agents collaborating on your order..."):
                    status, log, extra = process_order_pipeline(order, live_area)

                order["status"] = status
                order["log"] = log
                order.update(extra)
                st.session_state.orders.insert(0, order)

                if status == "Completed":
                    st.success(f"✅ Order **{order_id}** completed! Courier: {extra.get('courier','-')}, ETA: {extra.get('eta','-')}")
                    st.balloons()
                elif status == "Flagged for Review":
                    st.warning(f"⚠️ Order **{order_id}** flagged for manual review (risk score {extra.get('risk_score')}).")
                else:
                    st.error(f"❌ Order **{order_id}** could not be completed.")
        else:
            st.info("Fill in the form and submit to watch the agents process the order live.")

# ----------------------------------------------------------------------------
# PAGE: ORDER TRACKING
# ----------------------------------------------------------------------------
elif page == "📋 Order Tracking":
    st.markdown("<div class='section-title'>All Orders</div>", unsafe_allow_html=True)
    orders = st.session_state.orders

    if not orders:
        st.info("No orders yet. Create one from **🛒 New Order**.")
    else:
        f1, f2 = st.columns([1, 2])
        with f1:
            status_filter = st.multiselect("Filter by status", ["Completed", "Flagged for Review", "Failed"],
                                            default=["Completed", "Flagged for Review", "Failed"])
        with f2:
            search = st.text_input("Search by customer or order ID", "")

        filtered = [
            o for o in orders
            if o["status"] in status_filter
            and (search.lower() in o["customer"].lower() or search.lower() in o["order_id"].lower() or search == "")
        ]

        table_rows = [{
            "Order ID": o["order_id"],
            "Customer": o["customer"],
            "Product": o["product"],
            "Qty": o["quantity"],
            "Amount": f"${o['amount']:.2f}",
            "Status": o["status"],
            "Courier": o.get("courier", "—"),
            "ETA": o.get("eta", "—"),
            "Time": o["timestamp"].strftime("%d %b %H:%M"),
        } for o in filtered]

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Inspect Agent Log</div>", unsafe_allow_html=True)
        ids = [o["order_id"] for o in filtered]
        if ids:
            selected = st.selectbox("Select an order", ids)
            sel_order = next(o for o in orders if o["order_id"] == selected)
            badge_class = STATUS_COLORS.get(sel_order["status"], "badge-info")
            st.markdown(
                f"**{sel_order['order_id']}** — {sel_order['customer']} "
                f"<span class='badge {badge_class}'>{sel_order['status']}</span>",
                unsafe_allow_html=True,
            )
            for entry in sel_order["log"]:
                kind_badge = {"info": "badge-info", "success": "badge-success",
                              "warn": "badge-warn", "fail": "badge-fail"}[entry["kind"]]
                st.markdown(
                    f"<div class='log-line'><b>{entry['agent']}</b> "
                    f"<span class='badge {kind_badge}'>{entry['kind'].upper()}</span> "
                    f"<span style='color:#888'>{entry['time']}</span><br>{entry['message']}</div>",
                    unsafe_allow_html=True,
                )

# ----------------------------------------------------------------------------
# PAGE: AGENT NETWORK
# ----------------------------------------------------------------------------
elif page == "🕸️ Agent Network":
    st.markdown("<div class='section-title'>How the Agents Collaborate</div>", unsafe_allow_html=True)

    dot = """
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.25,0.15"];
        edge [color="#7c3aed", fontname="Helvetica", fontsize=10];

        Orchestrator [fillcolor="#ede9fe", color="#7c3aed"];
        Intake [fillcolor="#e0f2fe", color="#0284c7"];
        Inventory [fillcolor="#dcfce7", color="#16a34a"];
        Fraud [fillcolor="#fee2e2", color="#dc2626"];
        Payment [fillcolor="#fef3c7", color="#d97706"];
        Logistics [fillcolor="#e0e7ff", color="#4f46e5"];
        Notification [fillcolor="#fce7f3", color="#db2777"];

        Orchestrator -> Intake [label="1. validate"];
        Intake -> Inventory [label="2. check stock"];
        Inventory -> Fraud [label="3. score risk"];
        Fraud -> Payment [label="4. authorize"];
        Payment -> Logistics [label="5. ship"];
        Logistics -> Notification [label="6. notify"];
        Notification -> Orchestrator [label="feedback", style=dashed];
    }
    """
    st.graphviz_chart(dot, use_container_width=True)

    st.markdown("<div class='section-title'>Agent Roles</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, agent in enumerate(AGENTS):
        with cols[i % 2]:
            st.markdown(
                f"<div class='agent-card'><h4>{agent['icon']} {agent['name']}</h4>"
                f"<p>{agent['role']}</p></div>",
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# PAGE: ANALYTICS
# ----------------------------------------------------------------------------
elif page == "📊 Analytics":
    orders = st.session_state.orders
    st.markdown("<div class='section-title'>Deeper Analytics</div>", unsafe_allow_html=True)

    if not orders:
        st.info("Process a few orders first to unlock analytics.")
    else:
        df = pd.DataFrame(orders)

        c1, c2 = st.columns(2)
        with c1:
            prod_rev = df[df["status"] == "Completed"].groupby("product")["amount"].sum().reset_index()
            fig = px.bar(prod_rev, x="product", y="amount", title="Revenue by Product",
                         color="amount", color_continuous_scale="Purples")
            fig.update_layout(height=360, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            risk_df = df.dropna(subset=["risk_score"]) if "risk_score" in df.columns else pd.DataFrame()
            if not risk_df.empty:
                fig2 = px.histogram(risk_df, x="risk_score", nbins=20, title="Fraud Risk Score Distribution",
                                     color_discrete_sequence=["#db2777"])
                fig2.add_vline(x=55, line_dash="dash", line_color="red", annotation_text="Review threshold")
                fig2.update_layout(height=360, margin=dict(t=40))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No fraud scores recorded yet.")

        st.markdown("<div class='section-title'>Simulated Agent Performance (avg latency)</div>", unsafe_allow_html=True)
        perf = pd.DataFrame({
            "Agent": [a["name"].replace(" Agent", "") for a in AGENTS],
            "Avg Latency (ms)": [random.randint(80, 400) for _ in AGENTS],
        })
        fig3 = px.bar(perf, x="Agent", y="Avg Latency (ms)", color="Avg Latency (ms)",
                      color_continuous_scale="Teal")
        fig3.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<div class='section-title'>Raw Order Data</div>", unsafe_allow_html=True)
        display_df = df[["order_id", "customer", "product", "quantity", "amount", "status"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
