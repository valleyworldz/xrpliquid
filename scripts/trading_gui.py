#!/usr/bin/env python3
"""
🚀 MULTI-ASSET TRADING BOT GUI
===============================
Beautiful, user-friendly interface for the trading bot
- Modern dark theme with professional styling
- Real-time status updates and monitoring
- Easy token selection and configuration
- Live PnL tracking and performance metrics
- One-click start/stop functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import queue
import subprocess

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from newbotcode import get_available_tokens, SYMBOL_CFG
    NEWBOTCODE_AVAILABLE = True
except ImportError:
    NEWBOTCODE_AVAILABLE = False

class ModernTheme:
    """Modern dark theme colors and styles"""
    
    # Color palette
    BG_DARK = "#1e1e1e"
    BG_MEDIUM = "#2d2d2d"
    BG_LIGHT = "#3c3c3c"
    ACCENT_GREEN = "#4CAF50"
    ACCENT_RED = "#F44336"
    ACCENT_BLUE = "#2196F3"
    ACCENT_ORANGE = "#FF9800"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b0b0b0"
    BORDER = "#555555"
    
    # Fonts
    FONT_TITLE = ("Segoe UI", 18, "bold")
    FONT_HEADING = ("Segoe UI", 14, "bold")
    FONT_BODY = ("Segoe UI", 11)
    FONT_SMALL = ("Segoe UI", 9)
    FONT_MONO = ("Consolas", 10)

class TradingBotGUI:
    """Main GUI application for the trading bot"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.bot_process: Optional[subprocess.Popen] = None
        self.log_queue = queue.Queue()
        self.is_monitoring = False
        self.selected_token = "XRP"
        self.bot_stats = {
            "status": "Stopped",
            "pnl": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "uptime": "00:00:00",
            "account_value": 0.0
        }
        
        # Initialize UI element references
        self.log_text = None
        self.token_info_label = None
        self.status_labels = {}
        
        self.setup_window()
        self.create_styles()
        self.create_widgets()
        self.start_log_monitor()
        
    def setup_window(self):
        """Configure the main window"""
        self.root.title("🚀 Multi-Asset Trading Bot")
        self.root.geometry("1200x800")
        self.root.configure(bg=ModernTheme.BG_DARK)
        self.root.resizable(True, True)
        
        # Set window icon (if available)
        try:
            self.root.iconphoto(True, tk.PhotoImage(data=self.get_icon_data()))
        except:
            pass
            
    def get_icon_data(self):
        """Simple icon data for the window"""
        return """
        iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==
        """
    
    def create_styles(self):
        """Create custom ttk styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', 
                           background=ModernTheme.BG_DARK,
                           foreground=ModernTheme.TEXT_PRIMARY,
                           font=ModernTheme.FONT_TITLE)
        
        self.style.configure('Heading.TLabel',
                           background=ModernTheme.BG_DARK,
                           foreground=ModernTheme.TEXT_PRIMARY,
                           font=ModernTheme.FONT_HEADING)
        
        self.style.configure('Body.TLabel',
                           background=ModernTheme.BG_DARK,
                           foreground=ModernTheme.TEXT_SECONDARY,
                           font=ModernTheme.FONT_BODY)
        
        self.style.configure('Success.TLabel',
                           background=ModernTheme.BG_DARK,
                           foreground=ModernTheme.ACCENT_GREEN,
                           font=ModernTheme.FONT_BODY)
        
        self.style.configure('Error.TLabel',
                           background=ModernTheme.BG_DARK,
                           foreground=ModernTheme.ACCENT_RED,
                           font=ModernTheme.FONT_BODY)
        
        self.style.configure('Custom.TButton',
                           background=ModernTheme.ACCENT_BLUE,
                           foreground=ModernTheme.TEXT_PRIMARY,
                           font=ModernTheme.FONT_BODY,
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Start.TButton',
                           background=ModernTheme.ACCENT_GREEN,
                           foreground=ModernTheme.TEXT_PRIMARY,
                           font=ModernTheme.FONT_HEADING,
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Stop.TButton',
                           background=ModernTheme.ACCENT_RED,
                           foreground=ModernTheme.TEXT_PRIMARY,
                           font=ModernTheme.FONT_HEADING,
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Custom.TFrame',
                           background=ModernTheme.BG_MEDIUM,
                           borderwidth=1,
                           relief='solid')
        
        self.style.configure('Custom.TCombobox',
                           fieldbackground=ModernTheme.BG_LIGHT,
                           background=ModernTheme.BG_LIGHT,
                           foreground=ModernTheme.TEXT_PRIMARY,
                           borderwidth=1,
                           insertcolor=ModernTheme.TEXT_PRIMARY)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Multi-Asset Trading Bot", style='Title.TLabel')
        title_label.pack(pady=(10, 20))
        
        # Create main sections
        self.create_token_selection_section(main_frame)
        self.create_control_section(main_frame)
        self.create_status_section(main_frame)
        self.create_performance_section(main_frame)
        self.create_log_section(main_frame)
        
    def create_token_selection_section(self, parent):
        """Create token selection section"""
        section_frame = ttk.LabelFrame(parent, text="  🎯 Token Selection  ", style='Custom.TFrame')
        section_frame.pack(fill='x', pady=(0, 10), padx=5)
        
        # Token selection row
        token_frame = ttk.Frame(section_frame, style='Custom.TFrame')
        token_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(token_frame, text="Select Token:", style='Body.TLabel').pack(side='left', padx=(0, 10))
        
        # Token dropdown
        self.token_var = tk.StringVar(value=self.selected_token)
        self.token_combo = ttk.Combobox(token_frame, textvariable=self.token_var, 
                                       style='Custom.TCombobox', state='readonly', width=15)
        self.token_combo.pack(side='left', padx=(0, 20))
        self.load_available_tokens()
        
        # Token info display
        self.token_info_label = ttk.Label(token_frame, text="Loading token info...", style='Body.TLabel')
        self.token_info_label.pack(side='left', padx=(20, 0))
        
        # Bind token selection change
        self.token_combo.bind('<<ComboboxSelected>>', self.on_token_change)
        
        # Quick select buttons for popular tokens
        quick_frame = ttk.Frame(section_frame, style='Custom.TFrame')
        quick_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        ttk.Label(quick_frame, text="Quick Select:", style='Body.TLabel').pack(side='left', padx=(0, 10))
        
        popular_tokens = [
            ("💰 BTC", "BTC", "Bitcoin - Most stable"),
            ("🔷 ETH", "ETH", "Ethereum - DeFi leader"), 
            ("💎 XRP", "XRP", "Ripple - XRPL features"),
            ("🚀 SOL", "SOL", "Solana - High momentum"),
            ("🐕 DOGE", "DOGE", "Dogecoin - Meme power")
        ]
        
        for emoji_name, symbol, description in popular_tokens:
            btn = ttk.Button(quick_frame, text=emoji_name, style='Custom.TButton',
                           command=lambda s=symbol: self.quick_select_token(s))
            btn.pack(side='left', padx=2)
            
            # Add tooltip
            self.create_tooltip(btn, f"{symbol}: {description}")
    
    def create_control_section(self, parent):
        """Create bot control section"""
        section_frame = ttk.LabelFrame(parent, text="  🎮 Bot Control  ", style='Custom.TFrame')
        section_frame.pack(fill='x', pady=(0, 10), padx=5)
        
        control_frame = ttk.Frame(section_frame, style='Custom.TFrame')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Start button
        self.start_button = ttk.Button(control_frame, text="🚀 START TRADING", 
                                      style='Start.TButton', command=self.start_bot)
        self.start_button.pack(side='left', padx=(0, 10), ipadx=20, ipady=10)
        
        # Stop button
        self.stop_button = ttk.Button(control_frame, text="🛑 STOP TRADING", 
                                     style='Stop.TButton', command=self.stop_bot, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 20), ipadx=20, ipady=10)
        
        # Status indicator
        self.status_indicator = ttk.Label(control_frame, text="● Stopped", 
                                         style='Error.TLabel', font=ModernTheme.FONT_HEADING)
        self.status_indicator.pack(side='left', padx=(20, 0))
        
        # Configuration options
        config_frame = ttk.Frame(section_frame, style='Custom.TFrame')
        config_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Dry run checkbox
        self.dry_run_var = tk.BooleanVar()
        self.dry_run_check = ttk.Checkbutton(config_frame, text="🧪 Dry Run Mode (Safe Testing)", 
                                           variable=self.dry_run_var)
        self.dry_run_check.pack(side='left', padx=(0, 20))
        
        # Advanced options button
        ttk.Button(config_frame, text="⚙️ Advanced Options", style='Custom.TButton',
                  command=self.show_advanced_options).pack(side='left')
    
    def create_status_section(self, parent):
        """Create status monitoring section"""
        section_frame = ttk.LabelFrame(parent, text="  📊 Live Status  ", style='Custom.TFrame')
        section_frame.pack(fill='x', pady=(0, 10), padx=5)
        
        status_container = ttk.Frame(section_frame, style='Custom.TFrame')
        status_container.pack(fill='x', padx=10, pady=10)
        
        # Create status grid
        status_items = [
            ("Status:", "status", "❌ Stopped"),
            ("Account Value:", "account_value", "$0.00"),
            ("PnL:", "pnl", "$0.00"),
            ("Trades:", "trades", "0"),
            ("Win Rate:", "win_rate", "0%"),
            ("Uptime:", "uptime", "00:00:00")
        ]
        
        self.status_labels = {}
        
        for i, (label_text, key, default_value) in enumerate(status_items):
            row = i // 3
            col = (i % 3) * 2
            
            # Label
            ttk.Label(status_container, text=label_text, style='Body.TLabel').grid(
                row=row, column=col, sticky='w', padx=(0, 5), pady=2)
            
            # Value
            value_label = ttk.Label(status_container, text=default_value, style='Heading.TLabel')
            value_label.grid(row=row, column=col+1, sticky='w', padx=(0, 30), pady=2)
            self.status_labels[key] = value_label
    
    def create_performance_section(self, parent):
        """Create performance metrics section"""
        section_frame = ttk.LabelFrame(parent, text="  📈 Performance Metrics  ", style='Custom.TFrame')
        section_frame.pack(fill='x', pady=(0, 10), padx=5)
        
        perf_frame = ttk.Frame(section_frame, style='Custom.TFrame')
        perf_frame.pack(fill='x', padx=10, pady=10)
        
        # Performance cards
        self.create_performance_card(perf_frame, "💰 Total PnL", "$0.00", ModernTheme.ACCENT_GREEN, 0)
        self.create_performance_card(perf_frame, "📊 Win Rate", "0%", ModernTheme.ACCENT_BLUE, 1)
        self.create_performance_card(perf_frame, "⚡ Active Trades", "0", ModernTheme.ACCENT_ORANGE, 2)
        self.create_performance_card(perf_frame, "🎯 Best Trade", "$0.00", ModernTheme.ACCENT_GREEN, 3)
        
    def create_performance_card(self, parent, title, value, color, column):
        """Create a performance metric card"""
        card_frame = ttk.Frame(parent, style='Custom.TFrame')
        card_frame.grid(row=0, column=column, padx=5, pady=5, sticky='ew')
        parent.columnconfigure(column, weight=1)
        
        # Title
        title_label = ttk.Label(card_frame, text=title, style='Body.TLabel')
        title_label.pack(anchor='center', pady=(10, 5))
        
        # Value
        value_label = ttk.Label(card_frame, text=value, font=ModernTheme.FONT_HEADING)
        value_label.configure(foreground=color, background=ModernTheme.BG_MEDIUM)
        value_label.pack(anchor='center', pady=(0, 10))
        
        # Store reference for updates
        setattr(self, f"perf_{column}_label", value_label)
    
    def create_log_section(self, parent):
        """Create log display section"""
        section_frame = ttk.LabelFrame(parent, text="  📝 Live Logs  ", style='Custom.TFrame')
        section_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        # Log controls
        log_controls = ttk.Frame(section_frame, style='Custom.TFrame')
        log_controls.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Button(log_controls, text="🗑️ Clear Logs", style='Custom.TButton',
                  command=self.clear_logs).pack(side='left', padx=(0, 10))
        
        ttk.Button(log_controls, text="💾 Save Logs", style='Custom.TButton',
                  command=self.save_logs).pack(side='left', padx=(0, 10))
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="📄 Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side='right')
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(
            section_frame, 
            height=12, 
            bg=ModernTheme.BG_DARK,
            fg=ModernTheme.TEXT_PRIMARY,
            font=ModernTheme.FONT_MONO,
            insertbackground=ModernTheme.TEXT_PRIMARY,
            selectbackground=ModernTheme.ACCENT_BLUE,
            wrap=tk.WORD
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Configure log text tags for colored output
        self.setup_log_tags()
    
    def setup_log_tags(self):
        """Setup colored tags for log display"""
        self.log_text.tag_configure("INFO", foreground=ModernTheme.TEXT_PRIMARY)
        self.log_text.tag_configure("SUCCESS", foreground=ModernTheme.ACCENT_GREEN)
        self.log_text.tag_configure("WARNING", foreground=ModernTheme.ACCENT_ORANGE)
        self.log_text.tag_configure("ERROR", foreground=ModernTheme.ACCENT_RED)
        self.log_text.tag_configure("DEBUG", foreground=ModernTheme.TEXT_SECONDARY)
        
    def load_available_tokens(self):
        """Load available tokens for the dropdown"""
        try:
            if NEWBOTCODE_AVAILABLE:
                tokens = get_available_tokens()
            else:
                # Fallback token list
                tokens = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "MATIC", "UNI", "HYPE"]
            
            self.token_combo['values'] = tokens
            self.update_token_info()
            
        except Exception as e:
            self.log_message(f"⚠️ Could not load tokens: {e}", "WARNING")
            self.token_combo['values'] = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
    
    def update_token_info(self):
        """Update token information display"""
        if not hasattr(self, 'token_var') or not self.token_info_label:
            return
            
        token = self.token_var.get()
        
        # Token descriptions
        token_info = {
            "BTC": "Bitcoin - Most stable, highest liquidity, 40x leverage",
            "ETH": "Ethereum - DeFi leader, reliable trends, 25x leverage", 
            "XRP": "Ripple - XRPL features enabled, 20x leverage",
            "SOL": "Solana - High momentum, volatile, 25x leverage",
            "DOGE": "Dogecoin - Meme coin, sentiment-driven, 10x leverage",
            "ADA": "Cardano - Steady trends, moderate volatility",
            "AVAX": "Avalanche - Layer 1, high volatility",
            "LINK": "Chainlink - Oracle sector, tech-focused",
            "HYPE": "HYPE token - Mid-range price, 10x leverage"
        }
        
        info = token_info.get(token, f"{token} - Cryptocurrency trading")
        self.token_info_label.config(text=info)
    
    def quick_select_token(self, symbol):
        """Quick select a token"""
        self.token_var.set(symbol)
        self.selected_token = symbol
        self.update_token_info()
        self.log_message(f"🎯 Selected {symbol} for trading", "INFO")
    
    def on_token_change(self, event=None):
        """Handle token selection change"""
        self.selected_token = self.token_var.get()
        self.update_token_info()
        self.log_message(f"🎯 Token changed to {self.selected_token}", "INFO")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            tooltip.configure(bg=ModernTheme.BG_LIGHT)
            
            label = tk.Label(tooltip, text=text, bg=ModernTheme.BG_LIGHT, 
                           fg=ModernTheme.TEXT_PRIMARY, font=ModernTheme.FONT_SMALL,
                           wraplength=200)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            tooltip.after(3000, hide_tooltip)  # Auto-hide after 3 seconds
        
        widget.bind("<Enter>", show_tooltip)
    
    def start_bot(self):
        """Start the trading bot"""
        if self.bot_process and self.bot_process.poll() is None:
            messagebox.showwarning("Bot Running", "Trading bot is already running!")
            return
        
        # Confirmation dialog
        token = self.token_var.get()
        dry_run = self.dry_run_var.get()
        mode = "DRY RUN" if dry_run else "LIVE TRADING"
        
        result = messagebox.askyesno(
            "Start Trading Bot",
            f"Start {mode} with {token}?\n\n"
            f"Token: {token}\n"
            f"Mode: {mode}\n"
            f"This will begin automated trading!"
        )
        
        if not result:
            return
        
        try:
            # Build command
            cmd = [sys.executable, "newbotcode.py", "--symbol", token]
            if dry_run:
                cmd.append("--dry-run")
            
            # Start bot process
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.token_combo.config(state='disabled')
            
            self.status_indicator.config(text="● Running", style='Success.TLabel')
            self.bot_stats["status"] = "Running"
            
            # Start monitoring
            self.is_monitoring = True
            self.start_bot_monitoring()
            
            self.log_message(f"🚀 Started trading bot for {token} ({mode})", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Failed to start bot: {e}", "ERROR")
            messagebox.showerror("Start Failed", f"Failed to start trading bot:\n{e}")
    
    def stop_bot(self):
        """Stop the trading bot"""
        if not self.bot_process or self.bot_process.poll() is not None:
            messagebox.showwarning("Bot Not Running", "Trading bot is not running!")
            return
        
        result = messagebox.askyesno(
            "Stop Trading Bot", 
            "Stop the trading bot?\n\nThis will halt all automated trading."
        )
        
        if not result:
            return
        
        try:
            # Terminate bot process
            self.bot_process.terminate()
            self.bot_process.wait(timeout=10)
            
        except subprocess.TimeoutExpired:
            self.bot_process.kill()
            self.log_message("🔥 Force killed unresponsive bot process", "WARNING")
        
        except Exception as e:
            self.log_message(f"⚠️ Error stopping bot: {e}", "WARNING")
        
        finally:
            # Reset UI
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.token_combo.config(state='readonly')
            
            self.status_indicator.config(text="● Stopped", style='Error.TLabel')
            self.bot_stats["status"] = "Stopped"
            
            self.is_monitoring = False
            self.bot_process = None
            
            self.log_message("🛑 Trading bot stopped", "INFO")
    
    def start_bot_monitoring(self):
        """Start monitoring bot output and status"""
        def monitor_output():
            while self.is_monitoring and self.bot_process and self.bot_process.poll() is None:
                try:
                    line = self.bot_process.stdout.readline()
                    if line:
                        self.log_queue.put(("LOG", line.strip()))
                    time.sleep(0.1)
                except Exception as e:
                    self.log_queue.put(("ERROR", f"Monitor error: {e}"))
                    break
            
            if self.bot_process and self.bot_process.poll() is not None:
                self.log_queue.put(("STATUS", "Bot process ended"))
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()
    
    def start_log_monitor(self):
        """Start the log queue monitor"""
        def process_log_queue():
            try:
                while True:
                    msg_type, message = self.log_queue.get_nowait()
                    
                    if msg_type == "LOG":
                        self.process_bot_log(message)
                    elif msg_type == "STATUS":
                        self.log_message(message, "INFO")
                    elif msg_type == "ERROR":
                        self.log_message(message, "ERROR")
                        
            except queue.Empty:
                pass
            
            # Schedule next check
            self.root.after(100, process_log_queue)
        
        # Start the log monitor
        self.root.after(100, process_log_queue)
    
    def process_bot_log(self, message):
        """Process a log message from the bot"""
        # Extract log level and format message
        log_level = "INFO"
        display_message = message
        
        if "ERROR" in message or "❌" in message:
            log_level = "ERROR"
        elif "WARNING" in message or "⚠️" in message:
            log_level = "WARNING"
        elif "SUCCESS" in message or "✅" in message:
            log_level = "SUCCESS"
        elif "DEBUG" in message:
            log_level = "DEBUG"
        
        # Update status from specific log messages
        self.update_status_from_log(message)
        
        # Display in log
        self.log_message(display_message, log_level)
    
    def update_status_from_log(self, message):
        """Update bot status from log messages"""
        try:
            # Parse various status indicators from logs
            if "Account values" in message and "$" in message:
                # Extract account value
                import re
                value_match = re.search(r'Account Value: \$(\d+\.?\d*)', message)
                if value_match:
                    self.bot_stats["account_value"] = float(value_match.group(1))
                    self.status_labels["account_value"].config(text=f"${self.bot_stats['account_value']:.2f}")
            
            elif "ORDER SENT" in message:
                # Increment trade count
                self.bot_stats["trades"] += 1
                self.status_labels["trades"].config(text=str(self.bot_stats["trades"]))
            
            elif "PnL" in message and "$" in message:
                # Extract PnL
                import re
                pnl_match = re.search(r'PnL[:\s]*\$?(-?\d+\.?\d*)', message)
                if pnl_match:
                    self.bot_stats["pnl"] = float(pnl_match.group(1))
                    pnl_text = f"${self.bot_stats['pnl']:+.2f}"
                    self.status_labels["pnl"].config(text=pnl_text)
                    
                    # Update PnL color
                    color = ModernTheme.ACCENT_GREEN if self.bot_stats["pnl"] >= 0 else ModernTheme.ACCENT_RED
                    self.status_labels["pnl"].configure(foreground=color)
            
        except Exception as e:
            pass  # Ignore parsing errors
    
    def log_message(self, message, level="INFO"):
        """Add a message to the log display"""
        # If log_text isn't ready yet, just print to console
        if not self.log_text:
            print(f"[GUI] {message}")
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Insert message with appropriate tag
        self.log_text.insert(tk.END, formatted_message, level)
        
        # Auto-scroll if enabled
        if hasattr(self, 'auto_scroll_var') and self.auto_scroll_var.get():
            self.log_text.see(tk.END)
        
        # Limit log size (keep last 1000 lines)
        line_count = int(self.log_text.index(tk.END).split('.')[0])
        if line_count > 1000:
            self.log_text.delete("1.0", "100.0")
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.delete("1.0", tk.END)
        self.log_message("🗑️ Logs cleared", "INFO")
    
    def save_logs(self):
        """Save logs to file"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Logs"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get("1.0", tk.END))
                self.log_message(f"💾 Logs saved to {filename}", "SUCCESS")
        
        except Exception as e:
            self.log_message(f"❌ Failed to save logs: {e}", "ERROR")
    
    def show_advanced_options(self):
        """Show advanced configuration options"""
        # Create advanced options window
        adv_window = tk.Toplevel(self.root)
        adv_window.title("⚙️ Advanced Options")
        adv_window.geometry("500x400")
        adv_window.configure(bg=ModernTheme.BG_DARK)
        adv_window.resizable(False, False)
        
        # Make it modal
        adv_window.transient(self.root)
        adv_window.grab_set()
        
        # Center the window
        adv_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        # Advanced options content
        title = ttk.Label(adv_window, text="⚙️ Advanced Configuration", style='Title.TLabel')
        title.pack(pady=20)
        
        # Options frame
        options_frame = ttk.Frame(adv_window, style='Custom.TFrame')
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Add some advanced options
        options = [
            ("🎯 Confidence Threshold", "0.08", "Minimum confidence for trade signals"),
            ("💰 Position Size", "Auto", "Trade position sizing method"),
            ("⏰ Cycle Interval", "15s", "Trading cycle frequency"),
            ("🛡️ Risk Level", "Medium", "Risk management level"),
            ("📊 Max Leverage", "Auto", "Maximum leverage to use"),
        ]
        
        for i, (label, default, description) in enumerate(options):
            row_frame = ttk.Frame(options_frame, style='Custom.TFrame')
            row_frame.pack(fill='x', pady=5)
            
            ttk.Label(row_frame, text=label, style='Body.TLabel').pack(side='left')
            ttk.Label(row_frame, text=default, style='Heading.TLabel').pack(side='right')
            ttk.Label(row_frame, text=description, style='Body.TLabel', 
                     foreground=ModernTheme.TEXT_SECONDARY).pack(side='right', padx=(0, 10))
        
        # Note
        note_frame = ttk.Frame(adv_window, style='Custom.TFrame')
        note_frame.pack(fill='x', padx=20, pady=10)
        
        note_text = "💡 Advanced options are automatically optimized.\nModify newbotcode.py for custom settings."
        ttk.Label(note_frame, text=note_text, style='Body.TLabel', 
                 foreground=ModernTheme.TEXT_SECONDARY, justify='center').pack()
        
        # Close button
        ttk.Button(adv_window, text="✅ Close", style='Custom.TButton',
                  command=adv_window.destroy).pack(pady=20)
    
    def run(self):
        """Start the GUI application"""
        self.log_message("🚀 Trading Bot GUI started", "SUCCESS")
        self.log_message("💡 Select a token and click START TRADING to begin", "INFO")
        
        # Handle window close
        def on_closing():
            if self.bot_process and self.bot_process.poll() is None:
                if messagebox.askokcancel("Quit", "Trading bot is running. Stop bot and quit?"):
                    self.stop_bot()
                    self.root.destroy()
            else:
                self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the GUI main loop
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = TradingBotGUI()
        app.run()
    except Exception as e:
        print(f"❌ GUI Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
