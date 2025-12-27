dev:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8004

loadtest:
	uv run locust -f performance/locust.py --host http://127.0.0.1:8004 --processes 4

loadtest-direct:
	uv run locust -f performance/locust_direct.py --host http://127.0.0.1:8004 --processes 4