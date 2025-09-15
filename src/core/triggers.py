#!/usr/bin/env python3
"""
Trigger Management
=================

This module encapsulates TP/SL as objects with states to prevent race conditions
and state desync. All updates go through this module.
"""

import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.core.exchange import HyperliquidClient, OrderResult
from src.core.utils import align_price_to_tick


class TriggerState(Enum):
    """Trigger states"""
    PENDING = "pending"
    PLACED = "placed"
    VERIFIED = "verified"
    SHIFTED = "shifted"
    CANCELLED = "cancelled"
    FILLED = "filled"
    ERROR = "error"


class TriggerType(Enum):
    """Trigger types"""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TRAILING_STOP = "trailing"


@dataclass
class Trigger:
    """Trigger object with state management"""
    trigger_type: TriggerType
    price: float
    size: int
    order_id: Optional[str] = None
    state: TriggerState = TriggerState.PENDING
    created_time: float = None
    last_update: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()
        if self.last_update is None:
            self.last_update = time.time()
        if self.metadata is None:
            self.metadata = {}
            
    def update_state(self, new_state: TriggerState, metadata: Optional[Dict[str, Any]] = None):
        """Update trigger state"""
        self.state = new_state
        self.last_update = time.time()
        if metadata:
            self.metadata.update(metadata)
            
    def is_active(self) -> bool:
        """Check if trigger is active"""
        return self.state in [TriggerState.PLACED, TriggerState.VERIFIED, TriggerState.SHIFTED]
        
    def is_filled(self) -> bool:
        """Check if trigger was filled"""
        return self.state == TriggerState.FILLED
        
    def is_cancelled(self) -> bool:
        """Check if trigger was cancelled"""
        return self.state == TriggerState.CANCELLED


class TriggerManager:
    """Manages TP/SL triggers with state machine"""
    
    def __init__(self, exchange_client: HyperliquidClient, logger: Optional[logging.Logger] = None):
        self.exchange = exchange_client
        self.logger = logger or logging.getLogger(__name__)
        self.triggers: Dict[str, Trigger] = {}
        self.position_triggers: Dict[str, List[str]] = {}  # position_id -> trigger_ids
        
    def place_tp_sl_pair(self, symbol: str, is_long: bool, position_size: int,
                         entry_price: float, tp_price: float, sl_price: float,
                         position_id: str) -> Dict[str, str]:
        """Place TP/SL pair with atomic operation"""
        try:
            # Create trigger objects
            tp_trigger = Trigger(
                trigger_type=TriggerType.TAKE_PROFIT,
                price=tp_price,
                size=position_size
            )
            
            sl_trigger = Trigger(
                trigger_type=TriggerType.STOP_LOSS,
                price=sl_price,
                size=position_size
            )
            
            # Place TP trigger
            tp_result = self._place_single_trigger(symbol, is_long, tp_trigger)
            if not tp_result.success:
                self.logger.error(f"❌ Failed to place TP trigger: {tp_result.error}")
                return {}
                
            # Place SL trigger
            sl_result = self._place_single_trigger(symbol, not is_long, sl_trigger)
            if not sl_result.success:
                # Cancel TP if SL fails
                if tp_result.order_id:
                    self.exchange.cancel_order(tp_result.order_id)
                self.logger.error(f"❌ Failed to place SL trigger: {sl_result.error}")
                return {}
                
            # Store triggers
            tp_trigger.order_id = tp_result.order_id
            tp_trigger.update_state(TriggerState.PLACED)
            
            sl_trigger.order_id = sl_result.order_id
            sl_trigger.update_state(TriggerState.PLACED)
            
            # Register with position
            self.triggers[tp_result.order_id] = tp_trigger
            self.triggers[sl_result.order_id] = sl_trigger
            self.position_triggers[position_id] = [tp_result.order_id, sl_result.order_id]
            
            self.logger.info(f"✅ Placed TP/SL pair: TP={tp_result.order_id}, SL={sl_result.order_id}")
            
            return {
                'tp_oid': tp_result.order_id,
                'sl_oid': sl_result.order_id,
                'tp_price': tp_price,
                'sl_price': sl_price
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error placing TP/SL pair: {e}")
            return {}
            
    def _place_single_trigger(self, symbol: str, is_buy: bool, trigger: Trigger) -> OrderResult:
        """Place a single trigger order"""
        try:
            # Align price to tick
            tick_size = 0.0001  # XRP tick size
            aligned_price = align_price_to_tick(trigger.price, tick_size, "neutral")
            
            result = self.exchange.place_order(
                symbol=symbol,
                is_buy=is_buy,
                size=trigger.size,
                price=aligned_price,
                order_type="limit"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error placing trigger: {e}")
            return OrderResult(success=False, error=str(e))
            
    def verify_triggers(self, position_id: str) -> bool:
        """Verify triggers are still active on exchange"""
        if position_id not in self.position_triggers:
            return False
            
        trigger_ids = self.position_triggers[position_id]
        all_verified = True
        
        try:
            # Get open orders from exchange
            open_orders = self.exchange.get_open_orders()
            open_order_ids = {order.get('oid') for order in open_orders}
            
            for trigger_id in trigger_ids:
                if trigger_id not in self.triggers:
                    continue
                    
                trigger = self.triggers[trigger_id]
                
                if trigger_id in open_order_ids:
                    if trigger.state != TriggerState.VERIFIED:
                        trigger.update_state(TriggerState.VERIFIED)
                        self.logger.info(f"✅ Verified trigger {trigger_id}")
                else:
                    # Trigger not found on exchange
                    if trigger.state not in [TriggerState.FILLED, TriggerState.CANCELLED]:
                        trigger.update_state(TriggerState.ERROR, {'error': 'Not found on exchange'})
                        all_verified = False
                        self.logger.warning(f"⚠️ Trigger {trigger_id} not found on exchange")
                        
            return all_verified
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying triggers: {e}")
            return False
            
    def update_trailing_stop(self, position_id: str, new_sl_price: float) -> bool:
        """Update trailing stop with retry logic"""
        if position_id not in self.position_triggers:
            return False
            
        trigger_ids = self.position_triggers[position_id]
        sl_trigger_id = None
        
        # Find SL trigger
        for trigger_id in trigger_ids:
            if trigger_id in self.triggers:
                trigger = self.triggers[trigger_id]
                if trigger.trigger_type == TriggerType.STOP_LOSS:
                    sl_trigger_id = trigger_id
                    break
                    
        if not sl_trigger_id:
            self.logger.error("❌ No SL trigger found for position")
            return False
            
        sl_trigger = self.triggers[sl_trigger_id]
        
        # Cancel existing SL
        if sl_trigger.order_id:
            cancel_success = self.exchange.cancel_order(sl_trigger.order_id)
            if not cancel_success:
                self.logger.warning(f"⚠️ Failed to cancel SL trigger {sl_trigger.order_id}")
                # Continue anyway - might have been filled
                
        # Place new SL with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create new SL trigger
                new_sl_trigger = Trigger(
                    trigger_type=TriggerType.STOP_LOSS,
                    price=new_sl_price,
                    size=sl_trigger.size
                )
                
                # Place new trigger (assuming short position for SL)
                result = self._place_single_trigger("XRP", False, new_sl_trigger)
                
                if result.success:
                    # Update trigger
                    sl_trigger.price = new_sl_price
                    sl_trigger.order_id = result.order_id
                    sl_trigger.update_state(TriggerState.SHIFTED, {'attempt': attempt + 1})
                    
                    # Update trigger registry
                    self.triggers[result.order_id] = sl_trigger
                    if sl_trigger.order_id in self.triggers:
                        del self.triggers[sl_trigger.order_id]
                        
                    self.logger.info(f"✅ Updated trailing stop to {new_sl_price}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Failed to place new SL (attempt {attempt + 1}): {result.error}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        
            except Exception as e:
                self.logger.error(f"❌ Error updating trailing stop (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        self.logger.error("❌ Failed to update trailing stop after all retries")
        return False
        
    def cancel_all_triggers(self, position_id: str) -> bool:
        """Cancel all triggers for a position"""
        if position_id not in self.position_triggers:
            return True  # No triggers to cancel
            
        trigger_ids = self.position_triggers[position_id]
        all_cancelled = True
        
        for trigger_id in trigger_ids:
            if trigger_id in self.triggers:
                trigger = self.triggers[trigger_id]
                
                if trigger.order_id and trigger.is_active():
                    cancel_success = self.exchange.cancel_order(trigger.order_id)
                    if cancel_success:
                        trigger.update_state(TriggerState.CANCELLED)
                        self.logger.info(f"✅ Cancelled trigger {trigger_id}")
                    else:
                        all_cancelled = False
                        self.logger.warning(f"⚠️ Failed to cancel trigger {trigger_id}")
                        
        # Clean up
        if all_cancelled:
            del self.position_triggers[position_id]
            
        return all_cancelled
        
    def get_trigger_status(self, position_id: str) -> Dict[str, Any]:
        """Get status of all triggers for a position"""
        if position_id not in self.position_triggers:
            return {}
            
        trigger_ids = self.position_triggers[position_id]
        status = {}
        
        for trigger_id in trigger_ids:
            if trigger_id in self.triggers:
                trigger = self.triggers[trigger_id]
                status[trigger.trigger_type.value] = {
                    'state': trigger.state.value,
                    'price': trigger.price,
                    'order_id': trigger.order_id,
                    'created_time': trigger.created_time,
                    'last_update': trigger.last_update
                }
                
        return status
        
    def cleanup_filled_triggers(self):
        """Clean up filled triggers"""
        filled_triggers = []
        
        for trigger_id, trigger in self.triggers.items():
            if trigger.is_filled():
                filled_triggers.append(trigger_id)
                
        for trigger_id in filled_triggers:
            del self.triggers[trigger_id]
            
        # Clean up position mappings
        for position_id, trigger_ids in list(self.position_triggers.items()):
            active_triggers = [tid for tid in trigger_ids if tid in self.triggers]
            if not active_triggers:
                del self.position_triggers[position_id]
            else:
                self.position_triggers[position_id] = active_triggers
                
    def get_active_triggers(self) -> List[Trigger]:
        """Get all active triggers"""
        return [trigger for trigger in self.triggers.values() if trigger.is_active()] 