import { useState } from "react";
import DigitCanvas from "./components/DigitCanvas.jsx";

const MODES = {
  digits: {
    label: "Números",
    symbols: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  },
  vowels_lower: {
    label: "Vocales minúsculas",
    symbols: ["a", "e", "i", "o", "u"],
  },
  vowels_upper: {
    label: "Vocales mayúsculas",
    symbols: ["A", "E", "I", "O", "U"],
  },
};

export default function App() {
  const [status, setStatus] = useState("Listo");
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState("digits");

  // Símbolo objetivo (depende del modo)
  const [targetSymbol, setTargetSymbol] = useState("7");

  function newRandomTarget() {
    const symbols = MODES[mode].symbols;
    const n = Math.floor(Math.random() * symbols.length);
    setTargetSymbol(symbols[n]);
    setResult(null);
    setStatus("Listo");
  }

  function repeatTarget() {
    setResult(null);
    setStatus("Listo");
  }

  function onModeChange(nextMode) {
    setMode(nextMode);
    setTargetSymbol(MODES[nextMode].symbols[0]);
    setResult(null);
    setStatus("Listo");
  }

  const showEvaluation = result && (
    (result.target_symbol !== null && result.target_symbol !== undefined) ||
    (result.target_digit !== null && result.target_digit !== undefined)
  );
  const titleByMode = mode === "digits" ? "Escribe el número:" : "Escribe la vocal:";

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "24px auto", padding: 16 }}>
      <h1 style={{ margin: 0 }}>Práctica de escritura (MLP)</h1>
      <p style={{ marginTop: 8, color: "#444" }}>
        Selecciona el modo de ejercicio, dibuja el símbolo solicitado y presiona <b>Predecir</b>.
      </p>

      <div style={{ display: "flex", gap: 8, margin: "8px 0 14px 0", flexWrap: "wrap" }}>
        {Object.entries(MODES).map(([key, cfg]) => (
          <button
            key={key}
            onClick={() => onModeChange(key)}
            style={{
              padding: "8px 12px",
              borderRadius: 10,
              border: mode === key ? "1px solid #1f3d8f" : "1px solid #ccc",
              background: mode === key ? "#edf2ff" : "white",
              cursor: "pointer",
            }}
          >
            {cfg.label}
          </button>
        ))}
      </div>

      {/* Panel de objetivo */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "12px 0", flexWrap: "wrap" }}>
        <div style={{ fontSize: 18 }}>
          {titleByMode}{" "}
          <span style={{ fontSize: 28, fontWeight: 800, padding: "2px 10px", border: "1px solid #ddd", borderRadius: 10 }}>
            {targetSymbol}
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
          <DigitCanvas onStatus={setStatus} onResult={setResult} targetSymbol={targetSymbol} mode={mode} />

          <div style={{ marginTop: 12, color: "#666", fontSize: 14 }}>
            Estado: <b>{status}</b>
          </div>

          <div style={{ marginTop: 10, color: "#666", fontSize: 13 }}>
            Consejos: hazlo grande, centrado y con trazo claro.
          </div>

          <div style={{ marginTop: 10, color: "#335", fontSize: 13, background: "#f7f9ff", border: "1px solid #d8e3ff", borderRadius: 10, padding: 10 }}>
            Usa <b>Guardar muestra</b> cuando el dibujo represente bien al objetivo del modo actual. Luego podrás reentrenar la red con esas muestras reales.
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
                {result.symbol ?? result.digit ?? "—"}
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
                    Objetivo: <b>{result.target_symbol ?? result.target_digit}</b>
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
