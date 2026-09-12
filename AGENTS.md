# Project instructions

- This is a local Python/Streamlit UFC analytics application. Preserve the local SQLite betting journal and credential files.
- The user has requested that completed code changes be committed and pushed to `https://github.com/Tonys-Coding/UFC-Predicter.git`, including future changes in this project. After appropriate checks, commit and push to the configured upstream unless the user says otherwise. Never force-push or overwrite remote work.
- Keep `.env`, private keys, the personal SQLite database, scraped caches, and logs out of Git. The reproducible training code and nonpersonal validation report may be committed.
- All fight-derived predictors must use strictly earlier event dates. Keep same-day fights together in validation, and fit preprocessing and probability calibration only inside each training window.
- Verify model changes with leakage/chronology tests, relevant integration tests, and a fresh model run when the local historical dataset is available. Report calibration and performance honestly; fitted calibration does not imply perfect probabilities or profitable trades.
