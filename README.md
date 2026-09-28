# ClaimGuard AI — Insurance Claim Risk & Fraud Intelligence

ClaimGuard AI is a portfolio-grade, local ML decision-support prototype for insurance/warranty claim triage. It combines synthetic claim data, feature engineering, a Random Forest classifier, probability-based risk bands, NetworkX entity relationships, a Python HTTP API, and an investigator dashboard.

> **Responsible-use note:** risk scores prioritize human review. They do not prove fraud and this prototype must not be used for automated adverse decisions on real customers.

## Highlights

- 1,600 reproducible synthetic claims
- Random Forest risk classifier
- LOW / MEDIUM / HIGH probability-based triage
- Precision, recall, F1 and ROC-AUC evaluation
- Claim-level investigation view
- Customer → Claim → Device → Repair Center relationship graph
- Feature-importance risk signals
- Local REST-style HTTP API
- Responsive browser dashboard
- Windows one-click launcher

## Architecture

```text
Synthetic Claims
      ↓
Feature Engineering
      ↓
Random Forest
      ├──→ Hold-out Evaluation
      ├──→ Risk Probability
      └──→ Feature Importance
                ↓
          Python HTTP API
          ├── Summary
          ├── Claim Investigation
          ├── Model Metrics
          └── Entity Graph
                ↓
       HTML/CSS/JavaScript UI
```

## ML model

- Algorithm: Random Forest Classifier
- Estimators: 220
- Max depth: 9
- Min samples per leaf: 4
- Class weighting: balanced
- Stratified 75/25 train-test split
- Fixed seed for reproducibility

The current seeded synthetic test split reports:

| Metric | Result |
|---|---:|
| Accuracy | 73.75% |
| Precision | 68.18% |
| Recall | 65.22% |
| F1 | 66.67% |
| ROC-AUC | 79.44% |

These results describe the synthetic demonstration dataset only and are not production performance claims.

## Feature set

The model uses claim and behavioral signals including:

- claim amount
- customer claim count
- recent claims
- previous fraud flag
- repair frequency
- device age
- days since purchase
- entity-risk count

## Entity analysis

NetworkX builds a claim-centered investigative graph:

```text
Customer ── Claim ── Device
              │
              └──── Repair Center
```

The graph provides contextual relationships; it is not a separate fraud classifier.

## Running locally

### Windows

Double-click `run_claimguard.bat`.

Keep the terminal window open. The dashboard runs at:

`http://127.0.0.1:5000`

Do **not** open `demo.html` directly because the dashboard needs the local API.

### Manual

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

Health check:

`http://127.0.0.1:5000/api/health`

## API

- `GET /api/health`
- `GET /api/summary`
- `GET /api/model`
- `GET /api/claim/<claim_id>`
- `GET /api/graph/<claim_id>`

## Project structure

```text
├── app.py
├── demo.html
├── requirements.txt
├── test_api.py
├── run_claimguard.bat
├── start_server_only.bat
└── docs/
    ├── ARCHITECTURE.md
    ├── MODEL_CARD.md
    └── RUNNING.md
```

## Scope and limitations

This is a portfolio demonstration using synthetic data. It does not include real customer information, production storage, cloud deployment, fairness/drift monitoring, calibrated production probabilities, or automated claim decisions.

## Future work

- RAG over policy/claims documentation
- LLM-grounded investigator explanations
- LLM response evaluation and hallucination checks
- production database
- model/data drift monitoring
- authentication and audit logging
- cloud deployment

## Author

**Manu Shree M** — Computer Science, IIIT Pune
