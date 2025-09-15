from prometheus_client import start_http_server, Counter

trade_count = Counter('xrpbot_trades_total', 'Total number of trades executed by the bot')

def start_metrics_server(port=8000):
    start_http_server(port)

def increment_trade_count():
    trade_count.inc() 