# Security policy

## Reporting a vulnerability

Please disclose privately by emailing [senanur.cetin.work@gmail.com](mailto:senanur.cetin.work@gmail.com) with the subject `[smart-factory-app security]`. Include the affected version or commit, impact, and minimal safe reproduction steps. Do **not** disclose credentials, personal data, exploits against third-party systems, or sensitive logs in public GitHub issues.

This is a portfolio/research project with no guaranteed response time or supported security-release schedule.

## Use and deployment boundaries

The telemetry/dashboard demonstrates a predictive-maintenance research workflow using public/simulated data, not a live industrial control system. Model scores do not certify equipment safety. Do not connect directly to PLCs or production infrastructure without a separate engineering and security assessment.

Use environment variables or a dedicated secret manager for credentials. Do not commit .env files, data containing personal information, or private model/data artifacts.
