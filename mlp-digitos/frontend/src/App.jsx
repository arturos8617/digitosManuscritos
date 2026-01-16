import { useState } from "react";
import DigitCanvas from "./components/DigitCanvas.jsx";

export default function App() {
  const [status, setStatus] = useState("Listo");
  const [result, setResult] = useState(null);

  // Dígito objetivo (lo que el niño debe escribir)
  const [targetDigit, setTargetDigit] = useState(7);

  function newRandomTarget() {
    const n = Math.floor(Math.random() * 10);
    setTargetDigit(n);
    setResult(null);
    setStatus("Listo");
  }

  function repeatTarget() {
    setResult(null);
    setStatus("Listo");
  }

  const showEvaluation = result && result.target_digit !== null && result.target_digit !== undefined;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "24px auto", padding: 16 }}>
      <h1 style={{ margin: 0 }}>Práctica de escritura de dígitos (MLP)</h1>
      <p style={{ marginTop: 8, color: "#444" }}>
        Dibuja el número solicitado, centrado y grande. Luego presiona <b>Predecir</b>.
      </p>

      {/* Panel de objetivo */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "12px 0" }}>
        <div style={{ fontSize: 18 }}>
          Escribe el número:{" "}
          <span style={{ fontSize: 28, fontWeight: 800, padding: "2px 10px", border: "1px solid #ddd", borderRadius: 10 }}>
            {targetDigit}
          </span>
        </div>

        <button
          onClick={newRandomTarget}
          style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #222", background: "white", cursor: "pointer" }}
        >
          Nuevo número
        </button>

        <button
          onClick={repeatTarget}
          style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #ccc", background: "white", cursor: "pointer" }}
        >
          Repetir intento
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 16 }}>
        {/* Canvas */}
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
          <DigitCanvas onStatus={setStatus} onResult={setResult} targetDigit={targetDigit} />

          <div style={{ marginTop: 12, color: "#666", fontSize: 14 }}>
            Estado: <b>{status}</b>
          </div>

          <div style={{ marginTop: 10, color: "#666", fontSize: 13 }}>
            Consejos: hazlo grande, centrado y con trazo claro. Para “1” puedes agregar un pequeño gancho arriba (estilo MNIST).
          </div>
        </div>

        {/* Resultado */}
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
          <h3 style={{ marginTop: 0 }}>Resultado</h3>

          {!result ? (
            <div style={{ color: "#666" }}>Aún no hay predicción.</div>
          ) : (
            <div>
              <div style={{ fontSize: 36, fontWeight: 800, marginBottom: 8 }}>
                {result.digit}
              </div>

              <div style={{ marginBottom: 6 }}>
                Confianza: <b>{(result.confidence * 100).toFixed(1)}%</b>
              </div>

              <div style={{ marginBottom: 10 }}>
                Latencia: <b>{result.latency_ms.toFixed(1)} ms</b>
              </div>

              {/* Evaluación educativa */}
              {showEvaluation && (
                <div style={{ padding: 12, borderRadius: 12, border: "1px solid #eee", background: "#fafafa" }}>
                  <div style={{ marginBottom: 6 }}>
                    Objetivo: <b>{result.target_digit}</b>
                  </div>

                  <div style={{ marginBottom: 6 }}>
                    ¿Correcto?:{" "}
                    <b style={{ color: result.match ? "#0a7" : "#c20" }}>
                      {result.match ? "Sí" : "No"}
                    </b>
                  </div>

                  <div style={{ marginBottom: 6 }}>
                    Similitud: <b>{result.similarity_score != null ? `${result.similarity_score.toFixed(1)}%` : "—"}</b>
                  </div>

                  <div style={{ marginTop: 10 }}>
                    Retroalimentación:
                    <div style={{ marginTop: 6, fontWeight: 700 }}>
                      {result.feedback || "—"}
                    </div>
                  </div>
                </div>
              )}

              {/* Si tu backend todavía no envía estos campos */}
              {!showEvaluation && (
                <div style={{ marginTop: 10, fontSize: 13, color: "#666" }}>
                  Nota: el backend aún no está devolviendo <code>match</code>, <code>similarity_score</code> y <code>feedback</code>.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
