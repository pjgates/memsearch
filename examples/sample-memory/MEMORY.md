# MEMORY.md

## User Preferences
- Prefers dark mode in all editors and terminals
- Likes concise, to-the-point answers
- Primary languages: Python, TypeScript, Go
- Uses vim keybindings everywhere
- Timezone: Asia/Shanghai (UTC+8)

## Team & Contacts
- Alice Chen — Tech Lead, owns backend architecture, prefers async communication via Slack
- Bob Wang — Product Manager, weekly standup every Monday 10:00 AM
- Charlie Li — DevOps, manages Kubernetes clusters and CI/CD pipelines
- Diana Zhang — Frontend Lead, React/Next.js expert

## Project: E-commerce Platform Rewrite
- Codebase: monorepo at github.com/acme/shop-v2
- Backend: Python 3.12 + FastAPI + SQLAlchemy 2.0
- Frontend: Next.js 14 with App Router
- Database: PostgreSQL 16 (primary) + Redis 7 (caching + sessions)
- Search: Milvus for product semantic search
- Deployment: Kubernetes on AWS EKS, Terraform for infra
- CI/CD: GitHub Actions, deploy to staging on PR merge, production on tag

## Architecture Decisions
- ADR-001: Use event-driven architecture with Kafka for inter-service communication
- ADR-002: Adopt CQRS pattern for order service to handle read/write scaling independently
- ADR-003: Use Milvus for product search instead of Elasticsearch — better semantic matching
- ADR-004: JWT + refresh tokens for auth, 15min access / 7day refresh

## Code Conventions
- Python: ruff for linting, black for formatting, mypy strict mode
- TypeScript: ESLint + Prettier, strict tsconfig
- Commits: conventional commits (feat/fix/chore/docs)
- PRs: require 1 approval + passing CI, squash merge only
