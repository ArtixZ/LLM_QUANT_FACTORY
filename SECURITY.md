# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability, credential exposure or data leak.
Report it privately to:

- **Jiang Jingzhe**
- **contact@jiangjingzhe.com**

Include the affected version or commit, impact, reproduction conditions and a minimal proof that
does not contain licensed data, live credentials or destructive payloads. You should receive an
acknowledgement within seven days.

## Supported version

Security fixes target the latest `main` branch. Historical research snapshots and old tags may not
receive backports.

## Deployment boundary

The web services are local research control planes by default. Before exposing them beyond
loopback:

- set a strong `AUTOALPHA_SERVICE_TOKEN`;
- terminate TLS at a trusted reverse proxy;
- restrict network access and filesystem mounts;
- keep market data read-only where possible;
- store LLM credentials in the OS keychain or a secret manager and keep IBKR authentication
  inside the local TWS / Gateway session;
- use PostgreSQL and managed backups for multi-user deployments;
- review job concurrency and sandbox policies.

The repository must never contain `.env`, API keys, runtime SQLite databases, logs, market data or
private LLM transcripts.
