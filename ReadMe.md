# 🤖 Agentic OMS — Agentic AI Order Management System

> A multi-agent order processing platform where specialized AI agents — Intake, Inventory, Fraud Detection, Payment, Logistics, and Notification — collaborate under an Orchestrator to process customer orders end-to-end, built as an interactive Streamlit demo.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Demo-orange)

---

## 📖 About

**Agentic OMS** demonstrates how a team of autonomous AI agents can work together to handle the full lifecycle of an order — instead of a single monolithic system. Each agent owns one responsibility, reasons independently, and hands off to the next agent in the pipeline, coordinated by an **Orchestrator Agent**. The result is processed live in the browser with a real-time execution log, dashboards, and analytics.

This project is designed as a **visual, interactive proof-of-concept** for agentic AI architectures applied to order management / e-commerce operations.

---

## ✨ Features

- 🧭 **7 Collaborating Agents** — Orchestrator, Intake, Inventory, Fraud Detection, Payment, Logistics, Notification
- 🛒 **Live Order Pipeline** — submit an order and watch agents process it step-by-step in real time
- 📊 **Interactive Dashboard** — KPIs, order outcome breakdown, orders-over-time charts
- 📋 **Order Tracking** — searchable/filterable order table with a full per-order agent execution log
- 🕸️ **Agent Network Diagram** — visual graph of how agents hand off work to one another
- 📈 **Analytics** — revenue by product, fraud risk-score distribution, simulated agent latency
- 🎨 **Polished UI** — custom CSS, gradient header, status badges, animated live logs
- 🧠 **Simulated AI Reasoning** — each agent produces human-readable rationale for its decisions (stock checks, risk scoring, payment authorization, courier assignment, etc.)

---

## 🗂️ Agents Overview

| Agent | Role |
|---|---|
| 🧭 Orchestrator | Coordinates the pipeline, routes orders between agents, makes the final call |
| 📝 Intake | Validates order data — customer, product, quantity, address |
| 📦 Inventory | Checks stock levels and reserves units, or flags a backorder |
| 🛡️ Fraud Detection | Scores order risk and escalates suspicious orders for review |
| 💳 Payment | Authorizes and captures payment, handles declines |
| 🚚 Logistics | Selects courier/warehouse and estimates delivery |
| 📧 Notification | Sends confirmation, tracking, and status updates to the customer |

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — interactive web UI
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Plotly](https://plotly.com/python/) — interactive charts
- Graphviz (via Streamlit's built-in `st.graphviz_chart`) — agent network diagram

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/agentic-oms.git
cd agentic-oms
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run agentic_order_management.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) and open it automatically in your browser.

> ⚠️ Always launch with `streamlit run <file>.py` — running it with plain `python <file>.py` will not start the web server.

---

## 📁 Project Structure

```
agentic-oms/
├── agentic_order_management.py   # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🖼️ Screenshots

_Add screenshots or a screen recording of the Dashboard, New Order pipeline, and Agent Network pages here._

---

## 🗺️ Roadmap

- [ ] Real LLM-backed agent reasoning via the Anthropic API
- [ ] Persistent database backend (SQLite/Postgres) instead of session state
- [ ] Exportable order receipts (PDF)
- [ ] Multi-user auth and role-based dashboards
- [ ] Webhook/queue-based agent communication for a production-style architecture

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Built as a demo project showcasing agentic AI design patterns applied to order management systems.