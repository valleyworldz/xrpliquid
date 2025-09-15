import websocket
import json
import threading
import time
import random
from collections import deque

class WebSocketFeed:
    def __init__(self, url, on_message_callback=None):
        self.url = url
        self.ws = None
        self.on_message_callback = on_message_callback
        self.thread = None
        self.reconnect_interval = 2  # Initial reconnect interval in seconds
        self.max_reconnect_interval = 60 # Maximum reconnect interval
        self.reconnect_attempts = 0
        self.running = False
        self.subscriptions = [] # To store active subscriptions
        self.connected_event = threading.Event() # Event to signal successful connection

        # Throttling attributes for WebSocket messages
        self.message_timestamps = deque() # Stores timestamps of sent messages
        self.max_messages_per_minute = 100 # HL WS throttling limit
        self.throttle_interval = 60 # seconds

    def _throttle_check(self):
        current_time = time.time()
        # Remove timestamps older than the throttle interval
        while self.message_timestamps and current_time - self.message_timestamps[0] > self.throttle_interval:
            self.message_timestamps.popleft()

        if len(self.message_timestamps) >= self.max_messages_per_minute:
            # Calculate time to wait until a slot opens up
            time_to_wait = self.throttle_interval - (current_time - self.message_timestamps[0])
            print(f"WebSocket message throttle hit. Waiting for {time_to_wait:.2f} seconds.")
            time.sleep(time_to_wait)
            # After waiting, re-check (in case multiple waits are needed)
            self._throttle_check()
        self.message_timestamps.append(time.time())

    def on_message(self, ws, message):
        """
        Callback function for handling incoming WebSocket messages.
        """
        # print(f"Received WebSocket message: {message}")
        self.reconnect_interval = 2 # Reset reconnect interval on successful message
        self.reconnect_attempts = 0
        if self.on_message_callback:
            try:
                parsed_message = json.loads(message)
                self.on_message_callback(parsed_message) # Parse JSON and pass to callback
            except json.JSONDecodeError as e:
                print(f"Error decoding WebSocket message JSON: {e}. Message: {message}")
            except Exception as e:
                print(f"Error in on_message callback: {e}. Message: {message}")

    def on_error(self, ws, error):
        """
        Callback function for handling WebSocket errors.
        """
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """
        Callback function for handling WebSocket close events.
        """
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected_event.clear()
        if self.running:
            self._reconnect()

    def on_open(self, ws):
        """
        Callback function for handling WebSocket open events.
        """
        print("WebSocket connection opened.")
        self.connected_event.set() # Signal that connection is open
        # Resubscribe to all active subscriptions after reconnecting
        for sub_msg in self.subscriptions:
            self.ws.send(json.dumps(sub_msg))
            print(f"Resubscribed to: {sub_msg}")

    def _reconnect(self):
        """
        Handles adaptive reconnection with exponential backoff and jitter.
        """
        self.reconnect_attempts += 1
        delay = min(self.max_reconnect_interval, self.reconnect_interval * (2 ** (self.reconnect_attempts - 1)))
        jitter = random.uniform(0, delay * 0.1) # Add up to 10% jitter
        final_delay = delay + jitter
        print(f"Attempting to reconnect in {final_delay:.2f} seconds (attempt {self.reconnect_attempts})...")
        time.sleep(final_delay)
        self.connect()

    def connect(self):
        """
        Establishes and maintains the WebSocket connection in a separate thread.
        """
        if self.running:
            print(f"Connecting to WebSocket at: {self.url}")
            # websocket.enableTrace(True) # Uncomment for detailed WebSocket logging
            self.ws = websocket.WebSocketApp(self.url,
                                    on_open=self.on_open,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True # Allow main program to exit even if thread is running
            self.thread.start()

    def start(self):
        """
        Starts the WebSocket connection process.
        """
        self.running = True
        self.connect()
        self.connected_event.wait(timeout=10) # Wait for connection to be established
        if not self.connected_event.is_set():
            raise Exception("WebSocket connection failed to open in time.")

    def stop(self):
        """
        Closes the WebSocket connection and stops reconnection attempts.
        """
        self.running = False
        if self.ws:
            self.ws.close()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1) # Wait for thread to finish
            print("WebSocket disconnected.")

    def subscribe(self, subscription_type, coin=None):
        """
        Sends a subscription message to the WebSocket.
        Args:
            subscription_type (str): Type of subscription (e.g., "trades", "l2Book", "allMids").
            coin (str, optional): The coin symbol for the subscription (e.g., "BTC"). Required for "trades" and "l2Book".
        """
        msg = {"method": "subscribe", "subscription": {"type": subscription_type}}
        if coin:
            msg["subscription"]["coin"] = coin

        self._throttle_check() # Apply throttling before sending message
        if self.ws and self.connected_event.is_set():
            self.ws.send(json.dumps(msg))
            self.subscriptions.append(msg) # Store subscription for reconnection
            print(f"Sent subscription: {msg}")
        else:
            print("WebSocket not connected. Cannot send subscription.")
            self.subscriptions.append(msg) # Store for when connection is established


