
# Enhanced Error Handling System
import logging
import traceback
from typing import Optional, Callable

class EnhancedErrorHandler:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 10
        self.error_recovery_functions = {}
        
    def handle_error(self, error, context="", recovery_func=None):
        '''Handle errors with recovery mechanisms'''
        try:
            self.error_count += 1
            error_msg = f"Error in {context}: {str(error)}"
            logging.error(error_msg)
            
            # Execute recovery function if provided
            if recovery_func and callable(recovery_func):
                try:
                    recovery_func()
                except Exception as recovery_error:
                    logging.error(f"Recovery function failed: {recovery_error}")
            
            # Check if too many errors
            if self.error_count >= self.max_errors:
                logging.critical("Too many errors, initiating emergency shutdown")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error handler failed: {e}")
            return False
    
    def reset_error_count(self):
        '''Reset error counter'''
        self.error_count = 0
