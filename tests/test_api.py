"""
Integration tests for FastAPI endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from server.main import app


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["tasks_file_exists"])

    def test_search_tickers_endpoint(self):
        res = self.client.get("/api/tickers/search?q=MSFT")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(any(t["symbol"] == "MSFT" for t in data))

    def test_stock_data_endpoint(self):
        res = self.client.get("/api/stock/AAPL")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertTrue(len(data["history"]) > 0)

    def test_predict_endpoint(self):
        res = self.client.post("/api/predict", json={"ticker": "NVDA", "horizon": 14})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ticker"], "NVDA")
        self.assertIn("forecast", data)
        self.assertEqual(data["forecast"]["horizon"], 14)

    def test_backtest_endpoint(self):
        res = self.client.post("/api/backtest", json={"ticker": "NVDA", "test_days": 10})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("backtest", data)
        self.assertEqual(data["backtest"]["test_days"], 10)

    def test_tasks_endpoint(self):
        res = self.client.get("/api/tasks")
        self.assertEqual(res.status_code, 200)
        self.assertIn("content", res.json())

    def test_empty_ticker_validation(self):
        res_stock = self.client.get("/api/stock/%20")
        self.assertEqual(res_stock.status_code, 400)

        res_predict = self.client.post("/api/predict", json={"ticker": "", "horizon": 14})
        self.assertEqual(res_predict.status_code, 400)

    def test_frontend_serving(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"TimesFM", res.content)

    def test_compare_predictions_endpoint(self):
        res = self.client.post("/api/predict/compare", json={"tickers": ["NVDA", "AAPL"], "horizon": 14})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["horizon"], 14)
        self.assertIn("comparisons", data)
        self.assertEqual(len(data["comparisons"]), 2)
        for comp in data["comparisons"]:
            self.assertIn("normalized_pct_trajectory", comp)
            self.assertIn("expected_return_pct", comp)

    def test_export_endpoints(self):
        # CSV Export
        res_csv = self.client.get("/api/export/AAPL?format=csv&horizon=14")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers.get("content-type", ""))
        self.assertIn("Date,Ticker,Type", res_csv.text)

        # JSON Export
        res_json = self.client.get("/api/export/AAPL?format=json&horizon=14")
        self.assertEqual(res_json.status_code, 200)
        json_data = res_json.json()
        self.assertEqual(json_data["ticker"], "AAPL")
        self.assertIn("forecast", json_data)

    def test_risk_endpoint(self):
        res = self.client.get("/api/risk/NVDA?horizon=14")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ticker"], "NVDA")
        self.assertIn("risk_metrics", data)
        self.assertIn("var_95_pct", data["risk_metrics"])
        self.assertIn("risk_tier", data["risk_metrics"])

    def test_vercel_entrypoint(self):
        from api.index import app as vercel_app
        from fastapi.testclient import TestClient
        vercel_client = TestClient(vercel_app)
        res = vercel_client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "online")

        # Verify unified routing serves frontend root through serverless entrypoint
        res_root = vercel_client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn(b"TimesFM", res_root.content)

    def test_static_direct_routes(self):
        res_css = self.client.get("/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("text/css", res_css.headers.get("content-type", ""))

        res_js = self.client.get("/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("javascript", res_js.headers.get("content-type", ""))

    def test_task_creation_and_messaging_endpoints(self):
        # Test task creation
        res_create = self.client.post("/api/tasks/create", json={
            "description": "Automated verification test task",
            "from_role": "Tester",
            "to_role": "Lead Developer",
            "priority": "LOW"
        })
        self.assertEqual(res_create.status_code, 200)
        self.assertTrue(res_create.json()["success"])

        # Test message post
        res_msg = self.client.post("/api/tasks/message", json={
            "author": "Tester",
            "message": "Verification message test"
        })
        self.assertEqual(res_msg.status_code, 200)
        self.assertTrue(res_msg.json()["success"])

        # Verify tasks reflects update
        res_tasks = self.client.get("/api/tasks")
        self.assertEqual(res_tasks.status_code, 200)
        self.assertIn("Automated verification test task", res_tasks.json()["content"])
        self.assertIn("Verification message test", res_tasks.json()["content"])

    def test_readonly_data_engine_safety(self):
        from engine.data_loader import StockDataEngine
        # Initialize an engine with memory cache
        engine = StockDataEngine()
        data = engine.get_stock_data("AAPL", period="1mo", interval="1d")
        self.assertIn("history", data)
        self.assertIn("current_price", data)
        # Fetching second time hits in-memory cache
        cached = engine.get_stock_data("AAPL", period="1mo", interval="1d")
        self.assertEqual(len(cached["history"]), len(data["history"]))


if __name__ == "__main__":
    unittest.main()


