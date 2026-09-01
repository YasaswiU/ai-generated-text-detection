/**
 * Thin wrapper around the backend REST API.
 *
 * The backend URL is read from VITE_API_URL (see .env.example). It is never
 * hard-coded, so the same build works against a local dev server or the
 * deployed Render backend depending on how it was built/configured.
 */
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Calls POST /api/analyze.
 * @param {string} text
 * @param {string} language - "auto" | "en" | "te" | "hi"
 */
export async function analyzeText(text, language = "auto") {
  let response;
  try {
    response = await fetch(`${API_URL}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language }),
    });
  } catch (networkError) {
    throw new ApiError(
      "Unable to reach the analysis server. Please check your connection and try again.",
      0
    );
  }

  if (!response.ok) {
    let detail = "Unable to analyze the text right now. Please try again.";
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response body wasn't JSON; keep the generic message
    }
    throw new ApiError(detail, response.status);
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) {
    throw new ApiError("Health check failed.", response.status);
  }
  return response.json();
}

export { ApiError, API_URL };
