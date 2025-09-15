import logging

class DustPositionHandler:
    def __init__(self, config):
        self.config = config
        self.dust_threshold = config.get('trading', {}).get('dust_position_threshold', 0.001)
        self.ignore_dust = config.get('trading', {}).get('ignore_dust_positions', True)
        
    def is_dust_position(self, asset, size, price):
        """Check if position is dust"""
        position_value = size * price
        return position_value < self.dust_threshold
    
    def handle_dust_positions(self, positions):
        """Handle dust positions"""
        
        dust_positions = []
        valid_positions = []
        
        for pos in positions:
            asset = pos.get('asset')
            size = pos.get('size', 0)
            price = pos.get('price', 0)
            
            if self.is_dust_position(asset, size, price):
                dust_positions.append(pos)
                logging.info(f"Dust position detected: {asset} - {size} @ {price}")
            else:
                valid_positions.append(pos)
        
        if dust_positions and self.ignore_dust:
            logging.warning(f"Ignoring {len(dust_positions)} dust positions")
        
        return valid_positions, dust_positions
    
    def get_dust_summary(self, positions):
        """Get summary of dust positions"""
        
        dust_summary = {}
        
        for pos in positions:
            asset = pos.get('asset')
            size = pos.get('size', 0)
            price = pos.get('price', 0)
            value = size * price
            
            if self.is_dust_position(asset, size, price):
                if asset not in dust_summary:
                    dust_summary[asset] = {"count": 0, "total_value": 0}
                
                dust_summary[asset]["count"] += 1
                dust_summary[asset]["total_value"] += value
        
        return dust_summary 