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


if __name__ == "__main__":
    unittest.main()
