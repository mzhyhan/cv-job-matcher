# cv-job-matcher container image for the web demo.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="cv-job-matcher" \
      org.opencontainers.image.description="Match a CV against a job description: score the fit and suggest improvements" \
      org.opencontainers.image.source="https://github.com/mzhyhan/cv-job-matcher" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY cv_matcher ./cv_matcher
RUN pip install ".[web]"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "cv_matcher.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
