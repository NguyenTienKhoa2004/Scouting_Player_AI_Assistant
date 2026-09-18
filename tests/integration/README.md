# Integration tests

PostgreSQL tests create a disposable database, apply every migration, exercise
the real writers, and drop that database afterward. Point the test at the local
PostgreSQL service:

```powershell
$env:MATCHMIND_INTEGRATION_DATABASE_URL = "postgresql://matchmind:1234567@localhost:5433/matchmind"
python -m unittest tests.integration.test_silver_postgres -v
```

The configured user must be allowed to create and drop databases. Without the
environment variable these tests are skipped, so unit-only development does not
silently connect to PostgreSQL.
