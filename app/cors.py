def parse_cors_allowed_origins(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins or "null" in origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit HTTP(S) origins")
    if any(not origin.startswith(("http://", "https://")) for origin in origins):
        raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit HTTP(S) origins")
    return list(dict.fromkeys(origins))
