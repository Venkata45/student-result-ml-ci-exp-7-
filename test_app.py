import unittest
from app import app

class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"healthy", res.data)

    def test_predict_endpoint(self):
        payload = {"cgpa": 7.54, "placement_exam_marks": 40.0}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"prediction", res.data)

if __name__ == "__main__":
    unittest.main()
