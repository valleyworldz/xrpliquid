
def execute_order_with_fixes(self, symbol: str, quantity: float, price: float, side: str) -> Dict:
    """Enhanced order execution with comprehensive fixes"""
    
    # Check if token is halted
    if hasattr(self, 'halted_tokens') and symbol in self.halted_tokens:
        halt_time = self.halted_tokens[symbol]
        if datetime.now() - halt_time < timedelta(minutes=30):
            return {"success": False, "error": "Trading halted", "skip_token": True}
        else:
            del self.halted_tokens[symbol]
    
    # Get tick size and minimum order value
    tick_sizes = {
        "BTC": 0.5, "ETH": 0.05, "SOL": 0.01, "DOGE": 0.0001, "AVAX": 0.01,
        "DYDX": 0.001, "APE": 0.001, "OP": 0.001, "LTC": 0.01, "ARB": 0.001,
        "INJ": 0.001, "CRV": 0.001, "LDO": 0.001, "LINK": 0.001, "STX": 0.001,
        "CFX": 0.001, "SNX": 0.001, "WLD": 0.001, "YGG": 0.001, "TRX": 0.0001,
        "UNI": 0.001, "SEI": 0.001, "MATIC": 0.0001, "ATOM": 0.001, "DOT": 0.001,
        "ADA": 0.0001, "XRP": 0.0001, "SHIB": 0.00000001, "BCH": 0.01,
        "FIL": 0.001, "NEAR": 0.001, "ALGO": 0.0001, "VET": 0.00001,
        "ICP": 0.001, "FTM": 0.0001, "THETA": 0.001, "XLM": 0.0001,
        "HBAR": 0.0001, "CRO": 0.0001, "RUNE": 0.001, "GRT": 0.001,
        "AAVE": 0.01, "COMP": 0.01, "MKR": 0.5, "SUSHI": 0.001,
        "1INCH": 0.001, "ZRX": 0.0001, "BAL": 0.001, "YFI": 0.5,
        "UMA": 0.001, "REN": 0.0001, "KNC": 0.001, "BAND": 0.001,
        "NMR": 0.01, "MLN": 0.01, "REP": 0.001, "LRC": 0.0001,
        "OMG": 0.001, "ZEC": 0.01, "XMR": 0.01, "DASH": 0.01,
        "ETC": 0.01, "BTT": 0.00000001, "WIN": 0.00000001,
        "CHZ": 0.0001, "HOT": 0.00000001, "DENT": 0.00000001,
        "VTHO": 0.00000001, "STMX": 0.0001, "ANKR": 0.0001,
        "CKB": 0.00001, "NULS": 0.001, "RLC": 0.001, "PROM": 0.001,
        "VRA": 0.00000001, "COS": 0.0001, "CTXC": 0.0001, "CHR": 0.0001,
        "MANA": 0.0001, "SAND": 0.0001, "AXS": 0.001, "ENJ": 0.0001,
        "ALICE": 0.001, "TLM": 0.0001, "HERO": 0.0001, "DEGO": 0.001,
        "ALPHA": 0.001, "AUDIO": 0.001, "RARE": 0.001, "SUPER": 0.001,
        "AGIX": 0.0001, "FET": 0.0001, "OCEAN": 0.0001, "BICO": 0.001,
        "UNFI": 0.001, "ROSE": 0.0001, "ONE": 0.0001, "LINA": 0.0001
    }
    
    min_order_values = {symbol: 10.0 for symbol in tick_sizes.keys()}
    
    # Round price to tick size
    tick_size = tick_sizes.get(symbol, 0.001)
    if tick_size > 0:
        tick_count = price / tick_size
        rounded_tick_count = round(tick_count)
        price = rounded_tick_count * tick_size
    
    # Calculate minimum quantity for minimum order value
    min_value = min_order_values.get(symbol, 10.0)
    min_qty = min_value / price
    quantity = max(abs(quantity), min_qty)
    
    # Apply side
    if side.lower() == "sell":
        quantity = -quantity
    
    # Round quantity
    if symbol in ["BTC", "ETH"]:
        quantity = round(quantity, 4)
    elif symbol in ["SOL", "AVAX", "LTC"]:
        quantity = round(quantity, 2)
    else:
        quantity = round(quantity, 2)
    
    # Execute order
    try:
        result = self.api.place_order(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            price=price,
            order_type="market"
        )
        
        if result.get("success"):
            return {"success": True, "order_id": result.get("order_id")}
        else:
            error = result.get("error", "Unknown error")
            
            # Handle specific errors
            if "Trading is halted" in error:
                if not hasattr(self, 'halted_tokens'):
                    self.halted_tokens = {}
                self.halted_tokens[symbol] = datetime.now()
                return {"success": False, "error": "Trading halted", "skip_token": True}
            elif "minimum value" in error.lower():
                return {"success": False, "error": "Minimum value error"}
            elif "tick size" in error.lower():
                return {"success": False, "error": "Tick size error"}
            elif "liquidity" in error.lower() or "match" in error.lower():
                return {"success": False, "error": "Liquidity error"}
            else:
                return {"success": False, "error": error}
                
    except Exception as e:
        return {"success": False, "error": str(e)}
