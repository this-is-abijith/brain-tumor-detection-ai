import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState("");
  const [model, setModel] = useState("cnn");
  const [loading, setLoading] = useState(false);
  const [time, setTime] = useState("");
  const [modelUsed, setModelUsed] = useState("");

  const handleUpload = async () => {
    if (!file) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);

    try {
      setLoading(true);
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );

      setResult(response.data.prediction);
      setConfidence(response.data.confidence);
      setTime(response.data.time_ms);
      setModelUsed(response.data.model_used);
      setLoading(false);
    } catch (error) {
      console.error(error);
      alert("Prediction failed!");
      setLoading(false);
    }
  };

  const chartData = {
    labels: ["SVM (HOG)", "MobileNetV2"],
    datasets: [
      {
        label: "Accuracy (%)",
        data: [86, 92],
        backgroundColor: ["#36a2eb", "#4caf50"],
    },
  ],
};

  return (
    <div className="container">
      <div className="dashboard-card">
        <h1>Brain Tumor Detection</h1>
        <p className="subtitle">
          Comparative Analysis using CNN & SVM
        </p>

        {/* Top Section - Chart + Table */}
        <div className="top-section">
          <div className="chart-section">
            <Bar data={chartData} />
          </div>

          <div className="table-section">
            <h3>Model Comparison</h3>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>SVM (HOG)</td>
                  <td>86%</td>
                </tr>
                <tr>
                  <td>MobileNetV2 (Transfer Learning)</td>
                  <td>92%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Bottom Section - Landscape Layout */}
        <div className="bottom-section">
          <div className="left-panel">
            <select
              className="dropdown"
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="cnn">MobileNetV2 (CNN)</option>
              <option value="svm">SVM (HOG)</option>
            </select>

            <input
              type="file"
              onChange={(e) => {
                setFile(e.target.files[0]);
                setPreview(URL.createObjectURL(e.target.files[0]));
              }}
            />

            {preview && (
              <img
                src={preview}
                alt="Preview"
                className="preview-large"
              />
            )}

            <button className="predict-btn" onClick={handleUpload}>
              {loading ? "Predicting..." : "Predict"}
            </button>
          </div>

          <div className="right-panel">
            {result && (
              <div
                className={
                  result === "Tumor Detected"
                    ? "result tumor fade-in"
                    : "result normal fade-in"
                }
              >
                <h2>{result}</h2>
                <p>Confidence: {confidence}%</p>
                <p>Model Used: {modelUsed}</p>
                <p>Prediction Time: {time} ms</p>
              </div>
            )}
          </div>
        </div>

        <p className="disclaimer">
          ⚠ This system is for educational purposes only.
        </p>
      </div>
    </div>
  );
}
export default App;