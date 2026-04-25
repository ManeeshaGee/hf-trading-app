# ⚡ High-Frequency Trading (HFT) Simulation Platform

> A final-year undergraduate Software Engineering project — a cloud-native HFT simulation platform built with Next.js, Node.js, and Python AI microservices, hosted on Microsoft Azure. Multiple RL-powered trading agents compete in a real-time simulated market with an adaptive order matching engine.

![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=next.js&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=node.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Azure](https://img.shields.io/badge/Microsoft_Azure-0078D4?style=flat&logo=microsoftazure&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat&logo=mongodb&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [AI Integrations](#ai-integrations)
  - [RL Trading Agents (PPO / DQN)](#1-rl-trading-agents-ppo--dqn)
  - [Market Regime Detection](#2-market-regime-detection-hmm--lstm)
  - [Sentiment NLP Pipeline](#3-sentiment-nlp-pipeline)
  - [Anomaly Detection](#4-anomaly-detection--spoofing-detection)
- [Azure Cloud Integrations](#azure-cloud-integrations)
  - [Azure Event Hubs — Tick Ingestion](#5-azure-event-hubs--tick-ingestion)
  - [Azure Machine Learning — Model Training](#6-azure-machine-learning--model-training)
  - [Azure Blob Storage — Tick Data Lake](#7-azure-blob-storage--tick-data-lake)
  - [Azure Monitor + Logic Apps — Auto Evaluation](#8-azure-monitor--logic-apps--auto-evaluation)
- [Node.js ↔ Python Bridge](#nodejs--python-bridge)
- [Real-Time Simulation Loop](#real-time-simulation-loop)
- [Next.js Dashboard](#nextjs-dashboard)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Docker Compose](#docker-compose)
- [Azure Deployment](#azure-deployment)
- [Build Phases](#build-phases)
- [Contributing](#contributing)

---

## Overview

This platform simulates a real-world High-Frequency Trading environment where multiple AI-powered agents compete using different strategies — market making, momentum trading, statistical arbitrage, and noise trading. The system features a custom order matching engine, real-time AI inference, streaming market data ingestion via Azure Event Hubs, and automated model retraining via Azure Machine Learning.

**This is not a live trading app.** It is a simulation and research platform — every agent, order, and fill is synthetic. The goal is to study how AI strategies behave under different market regimes and to build production-grade cloud and AI engineering skills.

### Why Next.js + Node.js + Python?

| Layer | Technology | Reason |
|---|---|---|
| Frontend | Next.js 14 | SSR dashboard, Server Components, WebSocket client, streaming UI |
| Backend | Node.js + Express | Simulation orchestration, Socket.io, Azure SDK calls, REST API |
| AI | Python FastAPI | PyTorch, HuggingFace, hmmlearn — no JS equivalents exist |
| Cloud | Microsoft Azure | Event Hubs, Azure ML, Blob Storage, Monitor, Cache for Redis |

Node.js never does ML work directly. It calls the Python FastAPI microservice over HTTP for every AI decision and signal, then broadcasts results to Next.js via Socket.io.

---

## Key Features

- 🤖 **RL Trading Agents** — PPO and DQN agents with distinct strategies competing in the same simulated order book simultaneously
- 📈 **Adaptive Order Matching Engine** — price-time priority, lock-free ring buffer, sub-millisecond simulated matching
- 🌊 **Market Regime Detection** — Hidden Markov Model classifies the market as trending, mean-reverting, or high-volatility in real time
- 📰 **Sentiment NLP** — FinBERT processes simulated news headlines into a continuous sentiment signal fed to all agents
- 🚨 **Anomaly Detection** — Isolation Forest flags spoofing and wash trading patterns in live order flow
- ⚡ **Azure Event Hubs** — Kafka-compatible, high-throughput tick data streaming with per-symbol partition ordering
- 🧠 **Azure Machine Learning** — overnight RL model training on GPU compute clusters with a managed model registry
- 🗄️ **Azure Blob Storage** — Parquet tick data lake queryable via Azure Synapse Analytics for backtesting
- 📊 **Live Dashboard** — Next.js real-time PnL curves, order book depth, agent leaderboard, regime indicator
- 🔁 **Auto-Retraining Pipeline** — Azure Monitor triggers a Logic App workflow when the Sharpe ratio degrades

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Next.js 14 (Frontend)                   │
│  Dashboard · Order book · PnL chart · Agent leaderboard │
│  Server Components · API Routes · Socket.io-client      │
└─────────────────────┬───────────────────────────────────┘
                      │ REST + WebSocket
┌─────────────────────▼───────────────────────────────────┐
│               Node.js + Express (Backend)                │
│   Simulation engine · Socket.io · REST API              │
│   Azure SDK · Job orchestrator · MongoDB ODM            │
└──────────┬──────────────────────┬───────────────────────┘
           │ HTTP (FastAPI)        │ Azure SDK
┌──────────▼──────────┐  ┌────────▼──────────────────────┐
│  Python FastAPI      │  │      Microsoft Azure           │
│  AI Microservice     │  │                                │
│                      │  │  Event Hubs   (tick streaming) │
│  RL agents (PPO/DQN) │  │  Azure ML     (model training) │
│  Regime HMM + LSTM   │  │  Blob Storage (tick data lake) │
│  FinBERT sentiment   │  │  Cache for Redis               │
│  Isolation Forest    │  │  Monitor + Logic Apps          │
│  Azure ML SDK        │  │  Container Apps (deployment)   │
└──────────────────────┘  └────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│                      Data Layer                          │
│   MongoDB Atlas · Azure Cache for Redis · Blob Storage  │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14 (App Router), TailwindCSS, Recharts, Socket.io-client, TypeScript |
| **Backend** | Node.js, Express.js, Socket.io, Mongoose, BullMQ, Azure SDK for JavaScript |
| **AI / ML** | Python 3.11, FastAPI, PyTorch, Stable-Baselines3, hmmlearn, HuggingFace Transformers (FinBERT), scikit-learn |
| **Databases** | MongoDB Atlas, Azure Cache for Redis |
| **Streaming** | Azure Event Hubs (Kafka-compatible) |
| **Cloud ML** | Azure Machine Learning (GPU clusters, model registry, pipelines) |
| **Storage** | Azure Blob Storage (Parquet + Delta Lake) |
| **Analytics** | Azure Synapse Analytics (serverless SQL over Blob) |
| **Monitoring** | Azure Monitor, Application Insights, Azure Logic Apps |
| **Deployment** | Azure Container Apps, Azure Container Registry, GitHub Actions CI/CD |
| **Local Dev** | Docker Compose |

---

## Project Structure

```
hft-platform/
├── client/                              # Next.js 14 (App Router)
│   ├── app/
│   │   ├── layout.tsx                   # Root layout, global providers
│   │   ├── page.tsx                     # Landing page (SSR)
│   │   ├── dashboard/
│   │   │   └── page.tsx                 # Live simulation dashboard (Client Component)
│   │   ├── backtest/
│   │   │   └── page.tsx                 # Backtest results viewer (SSR)
│   │   ├── strategies/
│   │   │   └── page.tsx                 # Agent configuration panel
│   │   └── api/
│   │       ├── simulation/route.ts      # Start / stop simulation
│   │       └── agents/route.ts          # Fetch agent performance stats
│   ├── components/
│   │   ├── OrderBookChart.tsx           # Bid/ask depth visualisation
│   │   ├── PnLChart.tsx                 # Real-time PnL lines per agent
│   │   ├── AgentLeaderboard.tsx         # Agent ranking table
│   │   └── RegimeIndicator.tsx          # Current market regime badge
│   ├── hooks/
│   │   ├── useSimSocket.ts              # Socket.io live tick subscription
│   │   └── useAgentStats.ts             # SWR polling for episode stats
│   └── next.config.js
│
├── server/                              # Node.js + Express backend
│   ├── simulation/
│   │   ├── engine.js                    # Tick loop + Socket.io broadcaster
│   │   └── orderBook.js                 # JS order book (price-time priority)
│   ├── services/
│   │   ├── aiService.js                 # HTTP bridge → Python FastAPI
│   │   ├── eventHubService.js           # Azure Event Hubs producer/consumer
│   │   ├── blobService.js               # Azure Blob Storage tick archiving
│   │   └── azureMLService.js            # Trigger Azure ML training jobs
│   ├── routes/
│   │   ├── simulation.js                # POST /start, POST /stop, GET /status
│   │   ├── episodes.js                  # GET /episodes, GET /episodes/:id
│   │   └── agents.js                    # GET /agents, GET /agents/:id/pnl
│   ├── jobs/
│   │   └── retrainWorker.js             # BullMQ async retrain trigger
│   ├── socket/
│   │   └── index.js                     # Socket.io event handlers
│   └── middleware/
│       ├── auth.js                      # JWT verification
│       └── rateLimiter.js
│
├── ai-service/                          # Python FastAPI microservice
│   ├── routes/
│   │   ├── agents.py                    # POST /agents/{id}/act
│   │   ├── signals.py                   # POST /signals (regime + sentiment)
│   │   ├── anomaly.py                   # POST /anomaly/flag
│   │   └── training.py                  # POST /training/start
│   ├── models/
│   │   ├── ppo_agent.py                 # PyTorch PPO policy network
│   │   ├── dqn_agent.py                 # PyTorch DQN network
│   │   ├── regime_detector.py           # HMM + LSTM regime classifier
│   │   ├── sentiment_pipeline.py        # FinBERT sentiment scorer
│   │   └── anomaly_detector.py          # Isolation Forest order flow scanner
│   ├── azure/
│   │   └── ml_client.py                 # Azure ML SDK — submit training jobs
│   ├── requirements.txt
│   └── main.py
│
├── infra/                               # Infrastructure as Code
│   ├── bicep/                           # Azure Bicep templates
│   │   ├── main.bicep                   # Top-level deployment orchestration
│   │   ├── eventhub.bicep
│   │   ├── storage.bicep
│   │   ├── redis.bicep
│   │   └── containerApps.bicep
│   └── .github/
│       └── workflows/
│           ├── ci.yml                   # Lint + test on pull request
│           └── deploy.yml               # Build + push to Azure Container Registry
│
└── docker-compose.yml                   # Local development orchestration
```

---

## AI Integrations

### 1. RL Trading Agents (PPO / DQN)

Multiple agents with distinct strategies are trained via PyTorch and hosted by the Python FastAPI service. Node.js queries them every simulation tick to retrieve the next order action.

**State vector — what each agent observes per tick:**

```python
# ai-service/models/ppo_agent.py
state = {
    # Order book snapshot
    "bid_prices":   [...],     # top 5 bid price levels
    "ask_prices":   [...],     # top 5 ask price levels
    "bid_volumes":  [...],
    "ask_volumes":  [...],
    "spread":       float,
    "mid_price":    float,
    "imbalance":    float,     # (bid_vol - ask_vol) / total_vol

    # Market microstructure
    "vwap_1s":          float,
    "volatility_10s":   float,
    "trade_velocity":   float,   # trades per second
    "price_momentum":   float,

    # Upstream AI signals
    "regime":    int,    # 0=trending  1=mean-rev  2=volatile
    "sentiment": float,  # -1.0 to +1.0

    # Agent's own state
    "position":       int,
    "unrealised_pnl": float,
    "inventory_risk": float,
}

# Discrete action space
# 0=buy market  1=buy limit  2=hold
# 3=sell limit  4=sell market  5=cancel all
```

**PPO policy network:**

```python
import torch
import torch.nn as nn

class TradingPolicy(nn.Module):
    def __init__(self, state_dim=32, action_dim=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.actor  = nn.Linear(128, action_dim)  # action logits
        self.critic = nn.Linear(128, 1)           # value estimate

    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)

def compute_reward(pnl_delta, position, fill_price):
    # Penalise inventory build-up and transaction costs
    inventory_penalty = 0.001 * abs(position) ** 2
    transaction_cost  = 0.0001 * abs(fill_price)
    return pnl_delta - inventory_penalty - transaction_cost
```

**FastAPI endpoint — called by Node.js every tick:**

```python
# ai-service/routes/agents.py
from fastapi import APIRouter
import torch

router = APIRouter()
agents = {}  # loaded from Azure ML model registry on startup

@router.post("/agents/{agent_id}/act")
async def get_action(agent_id: str, payload: dict):
    state_vector = torch.tensor(payload["state"], dtype=torch.float32)
    policy = agents[agent_id]

    with torch.no_grad():
        logits, _ = policy(state_vector)
        action = torch.argmax(logits).item()

    return { "action": action }
```

---

### 2. Market Regime Detection (HMM + LSTM)

A Hidden Markov Model classifies the current market regime before each agent acts. The regime integer (0, 1, or 2) is injected into every agent's state vector as a conditioning signal.

```python
# ai-service/models/regime_detector.py
import hmmlearn.hmm as hmm
import numpy as np, pandas as pd

class RegimeDetector:
    """
    3 hidden states:
      0 = trending / momentum
      1 = mean-reverting / sideways
      2 = high-volatility / crisis
    """
    def __init__(self):
        self.hmm = hmm.GaussianHMM(
            n_components=3, covariance_type="full", n_iter=200
        )

    def fit(self, tick_history: list):
        features = self._extract_features(tick_history)
        self.hmm.fit(features)

    def predict(self, recent_ticks: list) -> int:
        features  = self._extract_features(recent_ticks)
        state_seq = self.hmm.predict(features)
        return int(state_seq[-1])  # 0, 1, or 2

    def _extract_features(self, ticks: list):
        prices   = np.array([t["mid_price"] for t in ticks])
        returns  = np.diff(np.log(prices + 1e-9))
        vol      = pd.Series(returns).rolling(20).std().fillna(0).values
        autocorr = pd.Series(returns).rolling(20).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        ).fillna(0).values
        return np.column_stack([returns[19:], vol[19:], autocorr[19:]])
```

---

### 3. Sentiment NLP Pipeline

FinBERT — a financial-domain fine-tuned BERT model — scores simulated news headlines. An exponentially-smoothed signal is fed into every agent's state vector as a continuous value between -1.0 and +1.0.

```python
# ai-service/models/sentiment_pipeline.py
from transformers import pipeline

class SentimentSignal:
    def __init__(self):
        self.model  = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=0   # GPU
        )
        self.signal = 0.0

    def score_headline(self, headline: str) -> float:
        result = self.model(headline)[0]
        if result["label"] == "positive": return  result["score"]
        if result["label"] == "negative": return -result["score"]
        return 0.0

    def update(self, headline: str, decay: float = 0.95):
        new_score   = self.score_headline(headline)
        # Exponential moving average — recent news matters more
        self.signal = decay * self.signal + (1 - decay) * new_score

    def get(self) -> float:
        return round(self.signal, 4)  # -1.0 to +1.0
```

---

### 4. Anomaly Detection — Spoofing Detection

An Isolation Forest runs on every order event and flags manipulative patterns such as quote stuffing, spoofing (large orders placed and immediately cancelled), and wash trading.

```python
# ai-service/models/anomaly_detector.py
from sklearn.ensemble import IsolationForest
import numpy as np

class OrderFlowAnomalyDetector:
    def __init__(self):
        self.model  = IsolationForest(
            n_estimators=200, contamination=0.01, random_state=42
        )
        self.buffer = []

    def extract_features(self, order_event: dict) -> list:
        return [
            order_event["size"],
            order_event["distance_from_mid"],    # price aggressiveness
            order_event["cancel_rate_1s"],        # key spoofing signal
            order_event["order_lifespan_ms"],
            order_event["agent_order_velocity"],  # orders per second
            order_event["size_vs_book_depth"],    # relative order size
        ]

    def flag(self, order_event: dict):
        features = self.extract_features(order_event)
        self.buffer.append(features)

        if len(self.buffer) >= 500:
            score      = self.model.score_samples([features])[0]
            is_anomaly = score < -0.5  # strong anomaly threshold
            return is_anomaly, float(score)

        return False, 0.0
```

---

## Azure Cloud Integrations

### 5. Azure Event Hubs — Tick Ingestion

Azure Event Hubs provides a Kafka-compatible, high-throughput streaming layer for tick data. Each symbol is assigned the same partition key, guaranteeing per-symbol ordering.

```js
// server/services/eventHubService.js
const { EventHubProducerClient, EventHubConsumerClient } = require('@azure/event-hubs');

const CONN_STRING = process.env.AZURE_EVENT_HUB_CONNECTION_STRING;
const HUB_NAME    = 'hft-tick-stream';

const producer = new EventHubProducerClient(CONN_STRING, HUB_NAME);

const publishTick = async (tick) => {
  const batch = await producer.createBatch({
    partitionKey: tick.symbol   // same symbol → same partition → ordering guaranteed
  });
  batch.tryAdd({
    body: {
      symbol:    tick.symbol,
      bid:       tick.bid,
      ask:       tick.ask,
      volume:    tick.volume,
      timestamp: new Date().toISOString(),
    }
  });
  await producer.sendBatch(batch);
};

const startConsuming = (onTick) => {
  const consumer = new EventHubConsumerClient('$Default', CONN_STRING, HUB_NAME);
  consumer.subscribe({
    processEvents: async (events) => {
      for (const event of events) {
        await onTick(event.body);  // → order book update → agent observation
      }
    },
    processError: async (err) => console.error('Event Hub error:', err),
  });
};

module.exports = { publishTick, startConsuming };
```

---

### 6. Azure Machine Learning — Model Training

Overnight training jobs are submitted to Azure ML GPU compute clusters. Trained models are stored in the Azure ML model registry and loaded by the Python FastAPI service at startup.

```python
# ai-service/azure/ml_client.py
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import os

client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"],
)

def submit_training_job(agent_id: str, data_path: str, strategy: str) -> str:
    job = command(
        code="./ai-service",
        command=(
            f"python train_ppo.py "
            f"--agent-id {agent_id} "
            f"--strategy {strategy} "
            f"--data-path {data_path} "
            f"--episodes 5000 "
            f"--learning-rate 3e-4"
        ),
        environment="pytorch-gpu@latest",   # Azure ML curated environment
        compute="gpu-cluster",              # NC-series GPU compute cluster
        experiment_name="hft-agent-training",
    )
    submitted = client.jobs.create_or_update(job)
    return submitted.name

def register_model(job_name: str, agent_id: str):
    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/model/",
        name=f"hft-agent-{agent_id}",
        type="custom_model",
    )
    client.models.create_or_update(model)

def load_latest_model(agent_id: str) -> str:
    model = client.models.get(name=f"hft-agent-{agent_id}", label="latest")
    return model.path
```

**Node.js triggers training via the Python service:**

```js
// server/services/azureMLService.js
const axios = require('axios');

const triggerRetraining = async (agentId, blobDataPath, strategy) => {
  const { data } = await axios.post(
    `${process.env.AI_SERVICE_URL}/training/start`,
    { agent_id: agentId, data_path: blobDataPath, strategy }
  );
  return data.job_name;
};

module.exports = { triggerRetraining };
```

---

### 7. Azure Blob Storage — Tick Data Lake

Every simulated tick is archived to Azure Blob Storage in Parquet format, partitioned by symbol and date. Azure Synapse Analytics serverless SQL runs backtesting queries directly over the Parquet files.

```js
// server/services/blobService.js
const { BlobServiceClient } = require('@azure/storage-blob');
const parquet = require('parquetjs');

const blobClient = BlobServiceClient.fromConnectionString(
  process.env.AZURE_STORAGE_CONNECTION_STRING
);
const container = blobClient.getContainerClient('hft-tick-data');

const archiveTicks = async (ticks, symbol, date) => {
  const schema = new parquet.ParquetSchema({
    symbol:    { type: 'UTF8' },
    bid:       { type: 'DOUBLE' },
    ask:       { type: 'DOUBLE' },
    volume:    { type: 'INT64' },
    timestamp: { type: 'UTF8' },
  });

  const writer = await parquet.ParquetWriter.openBuffer(schema);
  for (const tick of ticks) await writer.appendRow(tick);
  await writer.close();

  // Path: symbol=AAPL/date=2025-04-24/session.parquet
  const blobName = `symbol=${symbol}/date=${date}/session.parquet`;
  const blockBlob = container.getBlockBlobClient(blobName);
  await blockBlob.upload(writer.buffer, writer.buffer.length);
  console.log(`Archived ${ticks.length} ticks → ${blobName}`);
};

module.exports = { archiveTicks };
```

**Backtesting with Azure Synapse serverless SQL:**

```sql
-- Query the Parquet tick lake directly — no ETL required
SELECT
    symbol,
    AVG(ask - bid)   AS avg_spread,
    STDEV(ask - bid) AS spread_volatility,
    COUNT(*)         AS tick_count
FROM OPENROWSET(
    BULK 'https://hftstorage.blob.core.windows.net/hft-tick-data/symbol=AAPL/date=2025-*/**.parquet',
    FORMAT = 'PARQUET'
) AS ticks
GROUP BY symbol
ORDER BY avg_spread;
```

---

### 8. Azure Monitor + Logic Apps — Auto Evaluation

Node.js publishes custom metrics (Sharpe ratio, max drawdown, total PnL) to Azure Monitor after each episode. A Logic App workflow fires automatically when performance drops below threshold — logging an alert and triggering a retraining job.

```js
// server/services/monitorService.js
const publishMetrics = async (episodeResult) => {
  const { agentId, sharpe, maxDrawdown, totalPnl } = episodeResult;

  // POST custom metrics to Azure Monitor REST endpoint
  await fetch(
    `https://monitoring.azure.com${process.env.AZURE_RESOURCE_URI}/metrics`,
    {
      method:  'POST',
      headers: {
        'Authorization': `Bearer ${await getAzureToken()}`,
        'Content-Type':  'application/json',
      },
      body: JSON.stringify({
        time: new Date().toISOString(),
        data: {
          baseData: {
            metric:    'SharpeRatio',
            namespace: 'HFT/Simulation',
            dimNames:  ['AgentId'],
            series: [{
              dimValues: [agentId],
              min: sharpe, max: sharpe, sum: sharpe, count: 1,
            }]
          }
        }
      })
    }
  );
};

module.exports = { publishMetrics };
```

**Logic App — alert rule and auto-retrain trigger (Bicep):**

```bicep
// infra/bicep/logicApp.bicep
resource retrainAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'hft-sharpe-degraded'
  properties: {
    severity: 2
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [{
        name:            'LowSharpe'
        metricName:      'SharpeRatio'
        operator:        'LessThan'
        threshold:       '0.5'
        aggregation:     'Average'
        timeAggregation: 'PT5M'
      }]
    }
    actions: [{ actionGroupId: retrainActionGroup.id }]
  }
}
```

---

## Node.js ↔ Python Bridge

Node.js owns all Azure SDK calls and simulation orchestration. Python owns all ML inference. Every AI decision flows through HTTP to the FastAPI service.

```js
// server/services/aiService.js
const axios = require('axios');
const AI = process.env.AI_SERVICE_URL;  // http://ai-service:8000

// Get RL agent action — called every simulation tick per agent
const getAgentAction = async (agentId, stateVector) => {
  const { data } = await axios.post(`${AI}/agents/${agentId}/act`, {
    state: stateVector
  });
  return data.action;   // integer 0–5
};

// Get regime + sentiment — called once per tick for all agents
const getMarketSignals = async (recentTicks) => {
  const { data } = await axios.post(`${AI}/signals`, { ticks: recentTicks });
  return { regime: data.regime, sentiment: data.sentiment };
};

// Flag suspicious order — called on every order submission
const flagAnomaly = async (orderEvent) => {
  const { data } = await axios.post(`${AI}/anomaly/flag`, { order: orderEvent });
  return { isAnomaly: data.is_anomaly, score: data.score };
};

// Trigger Azure ML training job — proxied through Python ML SDK
const triggerRetraining = async (agentId, blobPath, strategy) => {
  const { data } = await axios.post(`${AI}/training/start`, {
    agent_id: agentId, data_path: blobPath, strategy
  });
  return data.job_name;
};

module.exports = { getAgentAction, getMarketSignals, flagAnomaly, triggerRetraining };
```

---

## Real-Time Simulation Loop

The tick loop runs in Node.js — it coordinates agents, matches orders, archives ticks to Azure Blob, and pushes live updates to Next.js via Socket.io.

```js
// server/simulation/engine.js
const { publishTick }    = require('../services/eventHubService');
const { archiveTicks }   = require('../services/blobService');
const { publishMetrics } = require('../services/monitorService');
const { getAgentAction, getMarketSignals, flagAnomaly } = require('../services/aiService');
const OrderBook = require('./orderBook');

class SimulationEngine {
  constructor(io) {
    this.io        = io;
    this.orderBook = new OrderBook();
    this.agents    = ['market_maker', 'momentum', 'arbitrageur', 'noise_trader'];
    this.running   = false;
    this.tickLog   = [];
  }

  async start(config) {
    this.running = true;
    let tick = 0;

    while (this.running && tick < config.maxTicks) {
      const marketState = this.orderBook.getState();

      // 1. Fetch AI signals once per tick (shared by all agents)
      const signals = await getMarketSignals(this.orderBook.recentTicks());

      // 2. Each agent decides in parallel
      const orders = await Promise.all(
        this.agents.map(async (agentId) => {
          const stateVec = this.buildStateVector(marketState, signals, agentId);
          const action   = await getAgentAction(agentId, stateVec);
          return this.actionToOrder(action, agentId);
        })
      );

      // 3. Anomaly check on every submitted order
      for (const order of orders) {
        const { isAnomaly } = await flagAnomaly(order);
        if (isAnomaly) order.flagged = true;
      }

      // 4. Run the matching engine
      orders.forEach(o => this.orderBook.submit(o));
      const fills = this.orderBook.match();

      // 5. Stream tick to Azure Event Hubs
      const tickData = { ...marketState, fills, timestamp: Date.now() };
      this.tickLog.push(tickData);
      await publishTick(tickData);

      // 6. Broadcast live update to Next.js dashboard
      this.io.emit('tick:update', {
        tick,
        orderBook:  marketState,
        fills,
        pnl:        this.getPnLSnapshot(),
        regime:     signals.regime,
        sentiment:  signals.sentiment,
        anomalies:  orders.filter(o => o.flagged).length,
      });

      tick++;

      // 7. Archive batch to Azure Blob Storage every 1000 ticks
      if (tick % 1000 === 0) {
        const date = new Date().toISOString().slice(0, 10);
        await archiveTicks(this.tickLog.splice(0), config.symbol, date);
      }

      await this.sleep(config.tickIntervalMs);
    }

    // 8. Episode complete — publish metrics to Azure Monitor
    await publishMetrics(this.computeEpisodeStats());
  }

  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
}

module.exports = SimulationEngine;
```

---

## Next.js Dashboard

```tsx
// client/app/dashboard/page.tsx
'use client';
import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';

const AGENTS  = ['market_maker', 'momentum', 'arbitrageur'];
const COLORS  = { market_maker: '#6366f1', momentum: '#14b8a6', arbitrageur: '#f59e0b' };
const REGIMES = ['Trending', 'Mean-reverting', 'High volatility'];

export default function Dashboard() {
  const [pnlHistory, setPnlHistory] = useState<any[]>([]);
  const [regime,     setRegime]     = useState<number | null>(null);
  const [sentiment,  setSentiment]  = useState<number>(0);
  const [anomalies,  setAnomalies]  = useState<number>(0);

  useEffect(() => {
    const socket = io(process.env.NEXT_PUBLIC_SOCKET_URL!);

    socket.on('tick:update', (data) => {
      setRegime(data.regime);
      setSentiment(data.sentiment);
      setAnomalies(prev => prev + data.anomalies);
      setPnlHistory(prev => [...prev.slice(-300), { tick: data.tick, ...data.pnl }]);
    });

    return () => socket.disconnect();
  }, []);

  return (
    <main style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem' }}>
        <span>Regime: <strong>{regime !== null ? REGIMES[regime] : '—'}</strong></span>
        <span>Sentiment: <strong>{sentiment.toFixed(3)}</strong></span>
        <span>Anomalies flagged: <strong>{anomalies}</strong></span>
      </div>

      <LineChart width={700} height={320} data={pnlHistory}>
        <XAxis dataKey="tick" />
        <YAxis />
        <Tooltip />
        <Legend />
        {AGENTS.map(a => (
          <Line key={a} dataKey={a} stroke={COLORS[a as keyof typeof COLORS]}
                dot={false} strokeWidth={1.5} />
        ))}
      </LineChart>
    </main>
  );
}
```

---

## Getting Started

### Prerequisites

- Node.js v18+
- Python 3.11+
- Docker + Docker Compose
- Azure CLI (`az login`) with an active subscription
- MongoDB Atlas account

### Clone and install

```bash
git clone https://github.com/your-username/hft-platform.git
cd hft-platform

# Backend dependencies
cd server && npm install

# Frontend dependencies
cd ../client && npm install

# Python AI service
cd ../ai-service
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run locally with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Next.js dashboard | http://localhost:3000 |
| Node.js API | http://localhost:5000 |
| Python FastAPI | http://localhost:8000 |
| FastAPI Swagger docs | http://localhost:8000/docs |

---

## Environment Variables

**`/server/.env`**

```env
# App
PORT=5000
NODE_ENV=development

# MongoDB
MONGO_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/hft

# JWT
JWT_SECRET=your_jwt_secret_here
JWT_EXPIRES_IN=7d

# Python AI microservice
AI_SERVICE_URL=http://ai-service:8000

# Azure Event Hubs
AZURE_EVENT_HUB_CONNECTION_STRING=Endpoint=sb://your-namespace.servicebus.windows.net/;SharedAccessKeyName=...
AZURE_EVENT_HUB_NAME=hft-tick-stream

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=hftstorage;...
AZURE_STORAGE_CONTAINER=hft-tick-data

# Azure Cache for Redis
REDIS_URL=rediss://<name>.redis.cache.windows.net:6380

# Azure Monitor
AZURE_SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_RESOURCE_GROUP=hft-rg
AZURE_RESOURCE_URI=/subscriptions/.../resourceGroups/hft-rg/providers/...
```

**`/ai-service/.env`**

```env
# Azure Machine Learning
AZURE_SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_RESOURCE_GROUP=hft-rg
AZURE_ML_WORKSPACE=hft-ml-workspace

# Azure Blob (training data paths)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=hftstorage;...
```

**`/client/.env.local`**

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_SOCKET_URL=http://localhost:5000
```

---

## Docker Compose

```yaml
# docker-compose.yml
services:
  client:
    build: ./client
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:5000
      - NEXT_PUBLIC_SOCKET_URL=http://localhost:5000

  server:
    build: ./server
    ports: ["5000:5000"]
    env_file: ./server/.env
    depends_on: [mongo, redis, ai-service]

  ai-service:
    build: ./ai-service
    ports: ["8000:8000"]
    env_file: ./ai-service/.env
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]   # GPU for FinBERT + PyTorch inference

  mongo:
    image: mongo:7
    ports: ["27017:27017"]
    volumes: ["mongo_data:/data/db"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

volumes:
  mongo_data:
```

---

## Azure Deployment

### 1. Provision infrastructure with Bicep

```bash
# Create resource group
az group create --name hft-rg --location eastus

# Deploy all Azure resources
az deployment group create \
  --resource-group hft-rg \
  --template-file infra/bicep/main.bicep \
  --parameters @infra/bicep/parameters.json
```

### 2. Build and push images to Azure Container Registry

```bash
az acr build --registry hftregistry --image hft-server:latest  ./server
az acr build --registry hftregistry --image hft-client:latest  ./client
az acr build --registry hftregistry --image hft-ai:latest      ./ai-service
```

### 3. Deploy to Azure Container Apps

```bash
az containerapp create \
  --name hft-server \
  --resource-group hft-rg \
  --environment hft-env \
  --image hftregistry.azurecr.io/hft-server:latest \
  --target-port 5000 \
  --ingress external

az containerapp create \
  --name hft-client \
  --resource-group hft-rg \
  --environment hft-env \
  --image hftregistry.azurecr.io/hft-client:latest \
  --target-port 3000 \
  --ingress external

az containerapp create \
  --name hft-ai \
  --resource-group hft-rg \
  --environment hft-env \
  --image hftregistry.azurecr.io/hft-ai:latest \
  --target-port 8000 \
  --ingress internal   # AI service is internal only
```

### Azure Services Reference

| Purpose | Azure Service |
|---|---|
| Real-time tick data streaming | Azure Event Hubs (Kafka-compatible) |
| GPU model training + registry | Azure Machine Learning |
| Parquet tick data lake | Azure Blob Storage |
| SQL backtesting over Blob | Azure Synapse Analytics (serverless) |
| In-memory order book state | Azure Cache for Redis |
| Metrics, logs, dashboards | Azure Monitor + Application Insights |
| Auto-retrain workflow | Azure Logic Apps |
| Serverless retrain trigger | Azure Functions |
| Container deployment | Azure Container Apps |
| Container image registry | Azure Container Registry |
| Infrastructure as Code | Azure Bicep |

---

## Build Phases

| Phase | Weeks | Goals |
|---|---|---|
| **Phase 1 — Core Engine** | 1–3 | Order book data structure, price-time priority matching, basic order types (market, limit, cancel), MongoDB schema |
| **Phase 2 — Simulation + Streaming** | 4–5 | Market simulation environment, Azure Event Hubs tick ingestion, Blob Storage archiving |
| **Phase 3 — RL Agents** | 6–8 | PPO/DQN policy networks, reward shaping, FastAPI bridge, Azure ML training job submission |
| **Phase 4 — AI Signals** | 9–10 | Regime HMM, FinBERT sentiment pipeline, Isolation Forest anomaly detector — all injected into agent state |
| **Phase 5 — Dashboard** | 11–12 | Next.js live dashboard, Socket.io real-time PnL, order book depth chart, agent leaderboard |
| **Phase 6 — Cloud + MLOps** | 13–14 | Azure Monitor custom metrics, Logic Apps auto-retrain, Synapse backtesting, Container Apps CI/CD pipeline |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is built as a final year undergraduate project for academic purposes.

---

> Built with Next.js · Node.js · Python · PyTorch · Microsoft Azure
