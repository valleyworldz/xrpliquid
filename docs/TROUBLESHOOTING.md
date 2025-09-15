
# Troubleshooting Guide: Hyperliquid Nova

## Introduction

This troubleshooting guide provides solutions to common issues you might encounter while setting up, running, or developing with Hyperliquid Nova. Before proceeding, please ensure you have followed the installation and configuration steps outlined in the `USER_GUIDE.md` and `DEVELOPER_GUIDE.md`.

## Common Issues and Solutions

### 1. Installation and Setup Issues

#### Issue: `git clone` fails or asks for credentials repeatedly.

*   **Symptom:** When trying to clone the repository, you get an error message like `Authentication failed` or `Username for 'https://github.com':` even after providing credentials.
*   **Possible Cause:** Incorrect GitHub token, token without sufficient permissions, or issues with Git credential helper configuration.
*   **Solution:**
    1.  **Verify Token:** Ensure the GitHub Personal Access Token (PAT) you are using is correct and has the `repo` scope enabled.
    2.  **Use Token in URL:** For quick cloning, you can embed the token directly in the URL (though less secure for public sharing):
        ```bash
        git clone https://ghp_YOUR_TOKEN@github.com/valleyworldz/hypeliquidOG.git
        ```
    3.  **Credential Helper:** If you configured a credential helper, ensure it's set up correctly. For example, to use the `store` helper:
        ```bash
        git config --global credential.helper store
        echo "https://ghp_YOUR_TOKEN@github.com" > ~/.git-credentials
        ```
        Make sure `~/.git-credentials` has the correct token and format.

#### Issue: `pip install -r requirements.txt` fails with dependency errors.

*   **Symptom:** Errors like `ModuleNotFoundError`, `Could not find a version that satisfies the requirement`, or compilation errors for certain packages.
*   **Possible Cause:** Missing system dependencies (e.g., for `PyQt5`), incompatible Python version, or corrupted virtual environment.
*   **Solution:**
    1.  **Python Version:** Ensure you are using Python 3.11 or higher. You can check with `python3.11 --version`.
    2.  **Virtual Environment:** Always use a virtual environment to avoid conflicts with system-wide Python packages:
        ```bash
        python3.11 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        ```
    3.  **System Dependencies (for PyQt5):** If you encounter errors related to PyQt5, you might need to install system-level dependencies. For Ubuntu/Debian:
        ```bash
        sudo apt-get update
        sudo apt-get install python3-pyqt5
        ```
        For other OS, refer to PyQt5 documentation.
    4.  **Clear Cache:** Sometimes, clearing pip's cache can resolve issues:
        ```bash
        pip cache purge
        ```

### 2. Application Runtime Issues

#### Issue: Application fails to start due to missing API keys.

*   **Symptom:** Error messages indicating `HYPERLIQUID_API_KEY` or `HYPERLIQUID_SECRET_KEY` are not found or are `None`.
*   **Possible Cause:** The `secrets.env` file is missing, incorrectly named, or environment variables are not being loaded.
*   **Solution:**
    1.  **Check `secrets.env`:** Ensure you have a file named `secrets.env` in the `configs/` directory of your project. Verify that the keys are correctly named (`HYPERLIQUID_API_KEY`, `HYPERLIQUID_SECRET_KEY`) and have valid values.
    2.  **Load Environment Variables:** If running directly, ensure your script or environment loads these variables. If using a tool like `python-dotenv`, make sure it's correctly configured. When running with Docker, ensure you pass these as environment variables (e.g., using `-e` flag or a `.env` file with `docker-compose`).
    3.  **Permissions:** Ensure the application has read permissions for the `configs/secrets.env` file.

#### Issue: Strategies are not executing or behaving as expected.

*   **Symptom:** Trading strategies are not placing orders, or their logic seems incorrect.
*   **Possible Cause:** Incorrect strategy configuration in `strategies_config.json`, issues with market data feed, or errors in strategy implementation.
*   **Solution:**
    1.  **Check `strategies_config.json`:** Verify that the `"enabled"` flag for your desired strategy is set to `true`. Double-check the `"params"` for the strategy to ensure they are correctly set and match the expected types (e.g., numbers for numerical parameters).
    2.  **Review Logs:** Check the application logs (`logs/` directory or console output) for any error messages or warnings related to strategy execution. The `core/utils/logger.py` module is designed to capture these.
    3.  **Market Data:** Ensure the `WebSocketFeed` and `HyperliquidAPI` are successfully connecting and receiving real-time market data. Look for connection errors in the logs.
    4.  **Debug Strategy:** Add `print()` statements or use a debugger within your strategy's `run` method (`core/strategies/*.py`) to inspect the `data` and `params` being passed and the logic's flow.

#### Issue: Risk management triggers unexpectedly or not at all.

*   **Symptom:** The `emergency_handler` is called without apparent reason, or critical risk thresholds are breached without intervention.
*   **Possible Cause:** Misconfigured risk parameters in `trading_params.json`, incorrect PnL calculation, or a bug in the `RiskManagement` module.
*   **Solution:**
    1.  **Review `trading_params.json`:** Check `max_risk_per_trade`, `max_daily_loss`, and other risk-related parameters. Ensure they are set to appropriate values for your risk tolerance.
    2.  **Verify PnL Calculation:** Inspect the `PnLAnalyzer` (`core/analytics/pnl_analyzer.py`) and how PnL is fed into the `RiskManagement` module. Ensure the PnL figures are accurate.
    3.  **Log Risk Events:** Enhance logging in `core/engines/risk_management.py` to output more details when risk checks are performed and when the `emergency_handler` is called. This can help pinpoint the exact condition that triggered it.
    4.  **Simulate Scenarios:** If possible, create test cases (in `tests/test_emergency_procedures.py`) that simulate various PnL and position scenarios to verify the risk management logic.

### 3. Development and CI/CD Issues

#### Issue: Unit tests (`pytest tests/`) are failing.

*   **Symptom:** `pytest` reports failures, often with `AssertionError` or `TypeError`.
*   **Possible Cause:** Recent code changes introduced bugs, test cases are outdated, or test environment is misconfigured.
*   **Solution:**
    1.  **Isolate Test:** Run the failing test individually (e.g., `pytest tests/test_strategy_execution.py::test_scalping_run_basic`) to get a more focused error message.
    2.  **Review Code Changes:** If tests were passing before, review your latest code changes for any logical errors or breaking changes.
    3.  **Mock Dependencies:** For unit tests, ensure external dependencies (like API calls) are properly mocked to isolate the code under test. This prevents network issues from causing test failures.
    4.  **Debug:** Use a Python debugger (e.g., `pdb`) to step through the failing test and the code it calls.

#### Issue: CI/CD pipeline (`ci_cd_pipeline.yml`) fails on GitHub Actions.

*   **Symptom:** GitHub Actions workflow run shows a red 'X' and fails at a specific step (e.g., `Install Dependencies`, `Run Unit Tests`, `Docker Build & Push`).
*   **Possible Cause:** Differences between local and CI environment, incorrect paths, missing secrets, or syntax errors in workflow file.
*   **Solution:**
    1.  **Check Logs:** The GitHub Actions workflow run provides detailed logs for each step. Examine the logs for the failing step to identify the exact error message.
    2.  **Environment Consistency:** Ensure your `requirements.txt` is up-to-date and includes all necessary dependencies. The Python version specified in the workflow (`python-version: 3.11`) should match your local development environment.
    3.  **Paths:** Verify that all paths in the workflow file (e.g., `pytest tests/`, `flake8 core/`) are correct relative to the repository root.
    4.  **Secrets:** If the `Docker Build & Push` step fails, ensure that your Docker registry credentials (e.g., `DOCKER_USERNAME`, `DOCKER_PASSWORD`) are correctly configured as GitHub Secrets in your repository settings.
    5.  **Linting/Security Scan:** If `flake8` or `bandit` steps fail, review the reported issues and fix them in your code. These tools enforce code quality and security best practices.

#### Issue: Docker build fails.

*   **Symptom:** The `docker build` command (either locally or in CI/CD) fails with errors.
*   **Possible Cause:** Incorrect `Dockerfile` syntax, missing files referenced in `Dockerfile`, or issues with base image.
*   **Solution:**
    1.  **Examine Dockerfile:** Review your `Dockerfile` for any syntax errors or incorrect commands. Ensure that all files and directories referenced (e.g., `COPY . /app`) actually exist.
    2.  **Context:** Make sure you are running `docker build` from the correct directory (the root of your project) so that all necessary files are in the build context.
    3.  **Base Image:** If the issue is with the base image, try a different version or a more stable one.
    4.  **Layer Caching:** Sometimes, issues can be resolved by forcing a rebuild without cache:
        ```bash
        docker build --no-cache -t hypeliquidog-nova:latest .
        ```

## Getting Further Help

If you have exhausted the solutions in this guide and are still facing issues, please consider the following:

*   **Consult Documentation:** Re-read the `USER_GUIDE.md`, `DEVELOPER_GUIDE.md`, and `API_DOCS.md` for any missed details.
*   **Review Codebase:** Carefully examine the relevant code files in the `core/` directory.
*   **Community Support:** If this were an open-source project, you would typically seek help from the community forums or issue trackers.
*   **Professional Support:** For enterprise-level deployments or critical issues, contact the Manus AI team for dedicated support.

---

*This document was generated by Manus AI.*

