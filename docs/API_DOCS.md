
# API Documentation: Hyperliquid Nova

## Introduction

This document provides detailed API documentation for Hyperliquid Nova, focusing on how external systems or advanced users can interact with its internal components programmatically. While Hyperliquid Nova primarily interacts with the Hyperliquid exchange API, it also exposes certain internal functionalities for integration and extension. This documentation covers the structure and usage of these internal APIs and how they can be leveraged.

## Internal API Endpoints (Conceptual)

Hyperliquid Nova is designed with modularity, and while it doesn't expose a public HTTP API by default in its current form, its components are designed to be callable internally. If a web-based GUI or external integrations were to be built, these conceptual endpoints would be exposed via a framework like FastAPI.

### 1. Strategy Management API

**Purpose:** To programmatically start, stop, or query the status of trading strategies.

**Conceptual Endpoint:** `/api/strategies`

**Methods:**

*   **`POST /api/strategies/start`**
    *   **Description:** Starts a specified trading strategy.
    *   **Request Body (JSON):**
        ```json
        {
          "strategy_name": "scalping",
          "token_symbol": "BTCUSD",
          "params": {
            "entry_deviation": 0.0001
          }
        }
        ```
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "message": "Scalping strategy started for BTCUSD",
          "strategy_id": "uuid-of-strategy-instance"
        }
        ```

*   **`POST /api/strategies/stop`**
    *   **Description:** Stops a running trading strategy.
    *   **Request Body (JSON):**
        ```json
        {
          "strategy_id": "uuid-of-strategy-instance"  // Or by name and symbol
        }
        ```
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "message": "Strategy stopped successfully"
        }
        ```

*   **`GET /api/strategies/status`**
    *   **Description:** Retrieves the status of all active strategies or a specific strategy.
    *   **Query Parameters (Optional):** `strategy_id`, `strategy_name`, `token_symbol`
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "active_strategies": [
            {
              "strategy_id": "uuid-of-strategy-instance",
              "name": "scalping",
              "token_symbol": "BTCUSD",
              "status": "running",
              "uptime_seconds": 3600
            }
          ]
        }
        ```

### 2. Trading Data & Analytics API

**Purpose:** To retrieve real-time trading data, performance metrics, and PnL information.

**Conceptual Endpoint:** `/api/data`

**Methods:**

*   **`GET /api/data/market/{symbol}`**
    *   **Description:** Fetches current market data for a given symbol.
    *   **Path Parameter:** `symbol` (e.g., `BTCUSD`)
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "symbol": "BTCUSD",
          "price": 70000.50,
          "volume_24h": 12345.67,
          "bid": 70000.00,
          "ask": 70001.00,
          "timestamp": 1678886400
        }
        ```

*   **`GET /api/data/positions`**
    *   **Description:** Retrieves all open trading positions.
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "positions": [
            {
              "symbol": "BTCUSD",
              "side": "long",
              "quantity": 0.01,
              "entry_price": 69500.00,
              "current_price": 70000.50,
              "unrealized_pnl": 5.00,
              "leverage": 10
            }
          ]
        }
        ```

*   **`GET /api/data/pnl`**
    *   **Description:** Fetches current Profit and Loss statistics.
    *   **Query Parameters (Optional):** `timeframe` (e.g., `daily`, `weekly`, `total`)
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "total_pnl": 1500.75,
          "daily_pnl": 75.20,
          "weekly_pnl": 300.50,
          "currency": "USD"
        }
        ```

*   **`GET /api/data/performance`**
    *   **Description:** Retrieves key performance indicators (KPIs) of the trading system.
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "win_rate": 0.65,
          "profit_factor": 1.8,
          "sharpe_ratio": 1.2,
          "max_drawdown": -0.03,
          "total_trades": 500
        }
        ```

### 3. Configuration Management API

**Purpose:** To dynamically update configuration parameters without restarting the application.

**Conceptual Endpoint:** `/api/config`

**Methods:**

*   **`GET /api/config/{param_name}`**
    *   **Description:** Retrieves the value of a specific configuration parameter.
    *   **Path Parameter:** `param_name` (e.g., `default_leverage`)
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "param_name": "default_leverage",
          "value": 10
        }
        ```

*   **`PUT /api/config/{param_name}`**
    *   **Description:** Updates the value of a specific configuration parameter.
    *   **Path Parameter:** `param_name`
    *   **Request Body (JSON):**
        ```json
        {
          "value": 20
        }
        ```
    *   **Response (JSON):**
        ```json
        {
          "status": "success",
          "message": "Parameter updated successfully",
          "param_name": "default_leverage",
          "new_value": 20
        }
        ```

## WebSocket API (Real-time Data Feed)

Hyperliquid Nova also utilizes a WebSocket connection for real-time, low-latency market data. While the primary connection is to the Hyperliquid exchange's WebSocket, an internal WebSocket server could be exposed for other applications to subscribe to processed data or internal events.

**Conceptual WebSocket URL:** `ws://localhost:8000/ws/data` (if exposed internally)

**Message Formats (Conceptual):**

*   **Subscription Request:**
    ```json
    {
      "type": "subscribe",
      "channels": ["trades", "orderbook"],
      "symbols": ["BTCUSD", "ETHUSD"]
    }
    ```

*   **Trade Update:**
    ```json
    {
      "type": "trade_update",
      "symbol": "BTCUSD",
      "price": 70000.50,
      "quantity": 0.001,
      "side": "buy",
      "timestamp": 1678886401
    }
    ```

*   **Order Book Update:**
    ```json
    {
      "type": "orderbook_update",
      "symbol": "BTCUSD",
      "bids": [["69999.50", "0.5"], ["69999.00", "1.2"]],
      "asks": [["70000.50", "0.8"], ["70001.00", "0.3"]],
      "timestamp": 1678886402
    }
    ```

## Authentication and Authorization

For any exposed internal APIs, robust authentication and authorization mechanisms would be implemented. This typically involves:

*   **API Keys:** Securely generated and managed API keys for client authentication.
*   **JWT (JSON Web Tokens):** For session management and stateless authentication in web-based scenarios.
*   **Role-Based Access Control (RBAC):** To define different levels of access for various users or services.

**Note:** The current implementation focuses on internal component interaction. For production deployment with exposed APIs, security considerations would be paramount and require careful implementation of these mechanisms.

## Error Handling

API responses will include clear error messages and appropriate HTTP status codes to indicate success or failure. Common error codes would include:

*   `200 OK`: Request successful.
*   `201 Created`: Resource successfully created.
*   `400 Bad Request`: Invalid request payload or parameters.
*   `401 Unauthorized`: Authentication failed or missing credentials.
*   `403 Forbidden`: Authenticated but not authorized to access the resource.
*   `404 Not Found`: Resource not found.
*   `500 Internal Server Error`: An unexpected error occurred on the server.

## SDK Usage (Conceptual)

While a formal SDK is not provided, the modular Python codebase allows developers to easily import and use the core components directly within their Python applications. For example:

```python
from core.api.hyperliquid_api import HyperliquidAPI
from core.strategies.scalping import Scalping
from core.engines.order_execution import OrderExecutionEngine

# Initialize API client
api_client = HyperliquidAPI(api_key="YOUR_API_KEY", secret_key="YOUR_SECRET_KEY")

# Get market data
btc_data = api_client.get_market_data("BTCUSD")
print(f"Current BTC price: {btc_data["price"]}")

# Initialize a strategy
scalping_strategy = Scalping()

# Define some mock data and parameters
data_for_strategy = {"current_price": btc_data["price"], "volume": btc_data["volume"]}
strategy_params = {"entry_deviation": 0.0001, "exit_deviation": 0.0002}

# Run the strategy (this would typically generate a signal)
scalping_strategy.run(data_for_strategy, strategy_params)

# Initialize order execution engine
order_executor = OrderExecutionEngine()

# Example: Place a mock order based on a hypothetical signal
mock_signal = {"symbol": "BTCUSD", "side": "buy", "quantity": 0.001, "price": btc_data["ask"]}
mock_risk_params = {"max_position_size": 0.01}
order_result = order_executor.execute_order(mock_signal, mock_risk_params)
print(f"Order result: {order_result}")
```

This direct import approach allows for maximum flexibility and integration within other Python-based trading systems or analytical tools.

---

*This document was generated by Manus AI.*

