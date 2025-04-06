import os
import time
import logging
import pandas as pd
from decimal import Decimal
from typing import Dict, List

from pydantic.v1 import Field
from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig


class MyPMMConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field("binance_paper_trade", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"))
    trading_pair: str = Field("ETH-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair in which the bot will place orders"))
    order_amount: Decimal = Field(0.01, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount (denominated in base asset)"))
    bid_spread: Decimal = Field(0.001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Base bid order spread (in percent)"))
    ask_spread: Decimal = Field(0.001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Base ask order spread (in percent)"))
    order_refresh_time: int = Field(15, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order refresh time (in seconds)"))
    price_type: str = Field("mid", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Price type to use (mid or last)"))
    candle_interval: str = Field("1m", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Candle interval (e.g., 1m, 5m, 1h)"))
    candle_length: int = Field(30, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "How many candles to use for NATR and RSI calculation"))
    candle_exchange: str = Field("binance", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange for candle data (e.g., binance)"))
    bid_spread_scalar: Decimal = Field(0.14, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bid spread scalar multiplier for NATR"))
    ask_spread_scalar: Decimal = Field(0.07, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Ask spread scalar multiplier for NATR"))
    use_volatility_spread: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Use volatility-based spread adjustment (True/False)"))
    use_trend_price_shift: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Use RSI-based trend price shifting (True/False)"))
    max_shift_spread: Decimal = Field(0.000001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximum price shift spread (in percent)"))
    trend_scalar: Decimal = Field(1, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trend scalar multiplier (-1 to follow trend, 1 to counter trend)"))
    # Inventory-related parameters
    target_inventory_ratio: Decimal = Field(0.5, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Target base asset ratio (0-1)"))
    inventory_scalar: Decimal = Field(1.0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Inventory price shift scalar"))
    use_inventory_price_shift: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Use inventory-based price shifting (True/False)"))
    # New trend protection parameters
    trend_threshold: Decimal = Field(0.10, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Price movement threshold to trigger trend protection (in percent)"))
    use_trend_protection: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Use trend protection and stop-loss features (True/False)"))
    trend_protection_multiplier: Decimal = Field(3.0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Increase order size when trend protection is active"))
    sell_skew_multiplier: Decimal = Field(2.0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Sell more in uptrends (multiplier)"))
    buy_skew_multiplier: Decimal = Field(1.5, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Buy more in downtrends (multiplier)"))


class UpdatedPMM(ScriptStrategyBase):
    create_timestamp = 0
    price_source = PriceType.MidPrice
    current_bid_spread = Decimal(0.001)
    current_ask_spread = Decimal(0.001)
    
    # Price shift related variables
    orig_price = Decimal(1)
    reference_price = Decimal(1)
    price_multiplier = Decimal(0)
    
    # Inventory shift related variables
    current_inventory_ratio = Decimal(0.5)
    inventory_delta = Decimal(0)
    inventory_multiplier = Decimal(0)
    
    # Trend protection variables
    initial_price = None
    price_change_pct = Decimal(0)
    trend_protection_active = False
    trend_direction = 0  # 1 for uptrend, -1 for downtrend, 0 for no significant trend
    base_order_amount = Decimal(0.01)  # Will be set from config
    
    @classmethod
    def init_markets(cls, config: MyPMMConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
        cls.price_source = (
            PriceType.LastTrade if config.price_type == "last" else PriceType.MidPrice
        )

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MyPMMConfig):
        super().__init__(connectors)
        self.config = config
        self.current_bid_spread = Decimal(self.config.bid_spread)
        self.current_ask_spread = Decimal(self.config.ask_spread)
        self.base_order_amount = Decimal(self.config.order_amount)
        
        # Initialize candles immediately in __init__
        self.logger().info(f"Initializing candles for {self.config.trading_pair} on {self.config.candle_exchange}, interval={self.config.candle_interval}")
        
        try:
            # Create candles feed
            self.candles = CandlesFactory.get_candle(CandlesConfig(
                connector=self.config.candle_exchange,
                trading_pair=self.config.trading_pair,
                interval=self.config.candle_interval,
                max_records=100
            ))
            
            # Start the candles feed immediately
            self.candles.start()
            self.logger().info("Candles initialized and started")
        except Exception as e:
            self.logger().error(f"Candle initialization failed: {str(e)}", exc_info=True)
            self.candles = None

    def on_start(self):
        # This is called after __init__, so we just log that we're starting
        self.logger().info("Strategy starting...")
        
        # If candles didn't initialize properly in __init__, try again
        if self.candles is None:
            self.logger().info("Retrying candle initialization...")
            try:
                # Create candles feed
                self.candles = CandlesFactory.get_candle(CandlesConfig(
                    connector=self.config.candle_exchange,
                    trading_pair=self.config.trading_pair,
                    interval=self.config.candle_interval,
                    max_records=100
                ))
                
                # Start the candles feed
                self.candles.start()
                self.logger().info("Candles initialized on retry and started")
            except Exception as e:
                self.logger().error(f"Candle initialization retry failed: {str(e)}", exc_info=True)

    def on_stop(self):
        if self.candles is not None:
            self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            
            # Update spreads and price shift based on volatility and trend if enabled
            self.update_market_parameters()
                
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.config.order_refresh_time + self.current_timestamp

    def get_candles_with_features(self):
        """Add NATR and RSI indicators plus spread calculations to candles dataframe"""
        if self.candles is None or not self.candles.ready or not hasattr(self.candles, 'candles_df') or self.candles.candles_df is None:
            return None
            
        # Make a copy to avoid modifying the original
        candles_df = self.candles.candles_df.copy()
        
        # Convert Decimal columns to float for pandas_ta compatibility
        for col in candles_df.columns:
            if len(candles_df) > 0 and isinstance(candles_df[col].iloc[0], Decimal):
                candles_df[col] = candles_df[col].astype(float)
        
        # Calculate NATR indicator
        try:
            # Calculate NATR indicator
            candles_df.ta.natr(length=self.config.candle_length, scalar=1, append=True)
            
            # Calculate RSI indicator
            candles_df.ta.rsi(length=self.config.candle_length, append=True)
            
            # Add moving average for trend detection
            candles_df.ta.sma(length=self.config.candle_length, append=True)
            
            # Calculate spread values in basis points
            bid_scalar = float(self.config.bid_spread_scalar)
            ask_scalar = float(self.config.ask_spread_scalar)
            
            candles_df['bid_spread_bps'] = candles_df[f"NATR_{self.config.candle_length}"] * bid_scalar * 10000
            candles_df['ask_spread_bps'] = candles_df[f"NATR_{self.config.candle_length}"] * ask_scalar * 10000
            
            return candles_df
        except Exception as e:
            self.logger().error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return None

    def update_market_parameters(self):
        """Update spreads based on NATR volatility and apply RSI-based trend price shift and inventory price shift"""
        candles_df = self.get_candles_with_features()
        
        if candles_df is not None and len(candles_df) > 0:
            try:
                # Get current price for trend protection calculation
                current_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
                
                # Initialize initial price if not set for trend protection
                if self.initial_price is None:
                    self.initial_price = current_price
                
                # Calculate price change percentage for trend protection
                if self.config.use_trend_protection:
                    self.price_change_pct = (current_price - self.initial_price) / self.initial_price
                    
                    # Detect significant trend and activate protection if needed
                    if abs(self.price_change_pct) >= self.config.trend_threshold:
                        self.trend_protection_active = True
                        self.trend_direction = 1 if self.price_change_pct > 0 else -1
                        self.logger().info(f"Trend protection activated: {self.trend_direction} (1=uptrend, -1=downtrend), change: {self.price_change_pct * 100:.2f}%")
                    else:
                        self.trend_protection_active = False
                        self.trend_direction = 0
                
                # Get the latest NATR value for spread calculation
                if f"NATR_{self.config.candle_length}" in candles_df.columns and self.config.use_volatility_spread:
                    latest_natr = candles_df[f"NATR_{self.config.candle_length}"].iloc[-1]
                    
                    if not pd.isna(latest_natr):
                        # Calculate new spreads - convert float to Decimal safely
                        bid_scalar = float(self.config.bid_spread_scalar)
                        ask_scalar = float(self.config.ask_spread_scalar)
                        
                        # Use string conversion to avoid precision issues when converting float to Decimal
                        self.current_bid_spread = Decimal(str(latest_natr * bid_scalar))
                        self.current_ask_spread = Decimal(str(latest_natr * ask_scalar))
                        
                        # Adjust spreads based on trend protection if active
                        if self.config.use_trend_protection and self.trend_protection_active:
                            if self.trend_direction == -1:  # Downtrend
                                # In downtrend, widen ask spread to avoid selling too cheap
                                # and tighten bid spread to buy at better prices
                                self.current_ask_spread *= Decimal('1.5')
                                self.current_bid_spread *= Decimal('0.8')
                            else:  # Uptrend
                                # In uptrend, tighten ask spread to sell quickly and lock in profits
                                # and widen bid spread to avoid buying at high prices
                                self.current_ask_spread *= Decimal('0.7')
                                self.current_bid_spread *= Decimal('2.0')
                                
                        self.logger().info(f"Updated volatility-based spreads - Bid: {self.current_bid_spread * 10000:.2f} bps, Ask: {self.current_ask_spread * 10000:.2f} bps")
                    else:
                        self.logger().warning("NATR value is NaN, using base spreads")
                        self.current_bid_spread = Decimal(self.config.bid_spread)
                        self.current_ask_spread = Decimal(self.config.ask_spread)
                else:
                    # Use base spreads if not using volatility-based spreads
                    self.current_bid_spread = Decimal(self.config.bid_spread)
                    self.current_ask_spread = Decimal(self.config.ask_spread)
                
                # Get original price first
                self.orig_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
                self.price_multiplier = Decimal(0)
                
                # Get RSI value for trend-based price shifting
                if f"RSI_{self.config.candle_length}" in candles_df.columns and self.config.use_trend_price_shift:
                    rsi = candles_df[f"RSI_{self.config.candle_length}"].iloc[-1]
                    
                    if not pd.isna(rsi):
                        # Calculate price multiplier based on RSI
                        # RSI > 50 means uptrend, < 50 means downtrend
                        # (rsi - 50) / 50 gives a range from -1 to 1
                        # Multiply by max_shift_spread to get the actual shift amount
                        # Multiply by trend_scalar to determine direction (-1 to follow trend, 1 to counter trend)
                        normalized_rsi = (rsi - 50) / 50
                        self.price_multiplier = Decimal(str(normalized_rsi)) * Decimal(self.config.max_shift_spread) * Decimal(self.config.trend_scalar)
                        
                        trend_price_shift = self.price_multiplier * self.orig_price
                        
                        self.logger().info(f"Updated trend-based price - RSI: {rsi:.2f}, Multiplier: {self.price_multiplier * 10000:.2f} bps, "
                                           f"Original: {self.orig_price}, Shift amount: {trend_price_shift}")
                    else:
                        self.logger().warning("RSI value is NaN, not applying trend shift")
                        self.price_multiplier = Decimal(0)
                
                # Calculate inventory-based price shift
                self.inventory_multiplier = Decimal(0)
                if self.config.use_inventory_price_shift:
                    try:
                        # Get base and quote asset balances
                        base_asset, quote_asset = self.config.trading_pair.split('-')
                        base_balance = self.connectors[self.config.exchange].get_balance(base_asset)
                        quote_balance = self.connectors[self.config.exchange].get_balance(quote_asset)
                        
                        # Convert base balance to quote value
                        base_value_in_quote = base_balance * self.orig_price
                        total_value = base_value_in_quote + quote_balance
                        
                        if total_value > Decimal('0'):
                            # Calculate current inventory ratio
                            self.current_inventory_ratio = base_value_in_quote / total_value
                            
                            # Adjust target ratio based on trend protection if active
                            target_ratio = self.config.target_inventory_ratio
                            if self.config.use_trend_protection and self.trend_protection_active:
                                if self.trend_direction == -1:  # Downtrend
                                    # In downtrend, reduce target ratio but still maintain some base asset
                                    target_ratio = max(Decimal('0.3'), self.config.target_inventory_ratio - Decimal('0.1'))
                                else:  # Uptrend
                                    # In uptrend, DECREASE target ratio to SELL more base asset and lock in profits
                                    target_ratio = max(Decimal('0.2'), self.config.target_inventory_ratio - Decimal('0.2'))
                            
                            # Calculate inventory delta (normalized difference from target)
                            delta = ((target_ratio - self.current_inventory_ratio) / 
                                    target_ratio)
                            
                            # Clamp delta to [-1, 1] range
                            self.inventory_delta = max(Decimal('-1'), min(Decimal('1'), Decimal(str(delta))))
                            
                            # Calculate inventory multiplier for price shift
                            self.inventory_multiplier = (self.inventory_delta * 
                                                        self.config.max_shift_spread * 
                                                        self.config.inventory_scalar)
                            
                            inventory_price_shift = self.inventory_multiplier * self.orig_price
                            
                            self.logger().info(f"Updated inventory-based price - Target ratio: {target_ratio}, "
                                              f"Current ratio: {self.current_inventory_ratio}, Delta: {self.inventory_delta}, "
                                              f"Multiplier: {self.inventory_multiplier * 10000:.2f} bps, "
                                              f"Shift amount: {inventory_price_shift}")
                        else:
                            self.logger().warning("Total portfolio value is zero, not applying inventory shift")
                            self.inventory_multiplier = Decimal(0)
                    except Exception as e:
                        self.logger().error(f"Error calculating inventory shift: {str(e)}", exc_info=True)
                        self.inventory_multiplier = Decimal(0)
                
                # Apply both price shifts to the reference price
                # Apply both shifts: (1 + trend_shift) * (1 + inventory_shift)
                self.reference_price = self.orig_price * (Decimal('1') + self.price_multiplier) * (Decimal('1') + self.inventory_multiplier)
                
                self.logger().info(f"Final reference price: {self.reference_price} (original: {self.orig_price}, "
                                  f"trend shift: {self.price_multiplier * 10000:.2f} bps, "
                                  f"inventory shift: {self.inventory_multiplier * 10000:.2f} bps)")
                
            except Exception as e:
                self.logger().error(f"Error updating market parameters: {str(e)}", exc_info=True)
                self.current_bid_spread = Decimal(self.config.bid_spread)
                self.current_ask_spread = Decimal(self.config.ask_spread)
                self.orig_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
                self.reference_price = self.orig_price
                self.price_multiplier = Decimal(0)
                self.inventory_multiplier = Decimal(0)
        else:
            # Use base values if we can't calculate indicators
            self.current_bid_spread = Decimal(self.config.bid_spread)
            self.current_ask_spread = Decimal(self.config.ask_spread)
            self.orig_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
            self.reference_price = self.orig_price
            self.price_multiplier = Decimal(0)
            self.inventory_multiplier = Decimal(0)
            self.logger().info(f"Using base parameters - Bid spread: {self.current_bid_spread}, Ask spread: {self.current_ask_spread}, Reference price: {self.reference_price}")

    def create_proposal(self) -> List[OrderCandidate]:
        connector = self.connectors[self.config.exchange]
        
        # Use the shifted reference price (which may be the same as the original price if trend shifting is disabled)
        ref_price = self.reference_price
        
        # Get best bid/ask from the order book
        best_bid = connector.get_price(self.config.trading_pair, False)
        best_ask = connector.get_price(self.config.trading_pair, True)

        # Calculate buy/sell prices using current spreads
        raw_buy_price = ref_price * (Decimal(1) - self.current_bid_spread)
        raw_sell_price = ref_price * (Decimal(1) + self.current_ask_spread)
        
        # Make sure our prices don't cross the order book
        buy_price = min(raw_buy_price, best_bid)
        sell_price = max(raw_sell_price, best_ask)
        
        # Set order amounts (default or adjusted for trend protection)
        buy_amount = self.config.order_amount
        sell_amount = self.config.order_amount
        
        # Adjust order size based on trend protection
        if self.config.use_trend_protection and self.trend_protection_active:
            if self.trend_direction == -1:  # Downtrend
                # Buy more in downtrend to average down cost basis
                buy_amount = self.config.order_amount * self.config.buy_skew_multiplier
            else:  # Uptrend
                # Sell more in uptrend to lock in profits
                sell_amount = self.config.order_amount * self.config.sell_skew_multiplier

        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=buy_amount,
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=sell_amount,
            price=sell_price
        )
        
        # If trend protection is active, possibly only place one-sided orders based on extreme conditions
        if self.config.use_trend_protection and self.trend_protection_active:
            if self.trend_direction == -1:  # Downtrend
                # In severe downtrend, if we have too much base asset, only sell
                if self.current_inventory_ratio > Decimal('0.8'):
                    self.logger().info("In severe downtrend with high base asset ratio - only placing sell order")
                    return [sell_order]
            elif self.trend_direction == 1:  # Uptrend
                # In strong uptrend, prioritize selling to lock in profits
                if self.current_inventory_ratio > Decimal('0.4'):  # If we have enough base asset
                    sell_order.amount = sell_amount * Decimal('1.5')  # Increase sell amount further
                    # If price has increased significantly, consider only selling
                    if self.price_change_pct > Decimal('0.15'):  # More than 15% increase
                        self.logger().info("In strong uptrend with >15% price increase - only placing sell order to lock in profits")
                        return [sell_order]  # Only place sell orders to lock in profits

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]):
        for order in proposal:
            if order.order_side == TradeType.BUY:
                self.buy(
                    connector_name=self.config.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
            else:
                self.sell(
                    connector_name=self.config.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} "
               f"{self.config.exchange} at {round(event.price, 2)}")
        
        # Reset initial price after significant fills to establish new trend baseline
        if self.config.use_trend_protection and event.amount > self.config.order_amount * Decimal('2'):
            current_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
            self.initial_price = current_price
            msg += f" - Reset trend baseline to {round(current_price, 2)}"
            
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []
        
        # Display header with basic info
        connector = self.connectors[self.config.exchange]
        current_price = connector.get_price_by_type(self.config.trading_pair, self.price_source)
        
        lines.extend([
            f"\n  {self.config.exchange} | {self.config.trading_pair} | Current Price: {current_price:.8f}",
            "========================================================================"
        ])
        
        # Display balances
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        # Display active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Active Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.append("\n  No active maker orders.")

        # Market metrics section
        best_bid = connector.get_price(self.config.trading_pair, False)
        best_ask = connector.get_price(self.config.trading_pair, True)
        best_bid_spread = (self.reference_price - best_bid) / self.reference_price
        best_ask_spread = (best_ask - self.reference_price) / self.reference_price
        bid_ask_spread = (best_ask - best_bid) / best_bid
        
        lines.extend([
            "\n  Market Metrics",
            "  " + "=" * 70,
            f"  Best Bid: {best_bid:.8f} | Best Ask: {best_ask:.8f} | Spread: {bid_ask_spread * 10000:.2f} bps"
        ])

        # Spread configuration section
        lines.extend([
            "\n  Spread Configuration",
            "  " + "=" * 70,
            f"  Bid Spread: {self.current_bid_spread * 10000:.2f} bps | Ask Spread: {self.current_ask_spread * 10000:.2f} bps",
            f"  Volatility Adjustment: {'ENABLED' if self.config.use_volatility_spread else 'DISABLED'}",
        ])
        
        if self.config.use_volatility_spread:
            lines.extend([
                f"  Bid Scalar: {self.config.bid_spread_scalar} | Ask Scalar: {self.config.ask_spread_scalar}"
            ])
        
        # Price adjustment section
        trend_price_shift = self.price_multiplier * self.orig_price
        inventory_price_shift = self.inventory_multiplier * self.orig_price
        total_price_shift = self.reference_price - self.orig_price
        
        lines.extend([
            "\n  Price Adjustment",
            "  " + "=" * 70,
            f"  Original Price: {self.orig_price:.8f} | Final Reference Price: {self.reference_price:.8f}",
            f"  Total Price Shift: {total_price_shift:.8f} ({(total_price_shift/self.orig_price) * 10000:.2f} bps)"
        ])
        
        if self.config.use_trend_price_shift:
            lines.extend([
                f"  Trend Adjustment: ENABLED",
                f"  Trend Scalar: {self.config.trend_scalar} | Trend Shift: {trend_price_shift:.8f} ({self.price_multiplier * 10000:.2f} bps)"
            ])
        
        if self.config.use_inventory_price_shift:
            lines.extend([
                f"  Inventory Adjustment: ENABLED",
                f"  Target Ratio: {self.config.target_inventory_ratio} | Current Ratio: {self.current_inventory_ratio:.4f}",
                f"  Inventory Delta: {self.inventory_delta:.4f} | Inventory Shift: {inventory_price_shift:.8f} ({self.inventory_multiplier * 10000:.2f} bps)"
            ])
        
        # Trend protection section
        if self.initial_price is not None:
            lines.extend([
                "\n  Trend Protection",
                "  " + "=" * 70,
                f"  Initial Price: {self.initial_price:.8f} | Current Change: {self.price_change_pct * 100:.2f}%",
                f"  Protection Active: {self.trend_protection_active} | Direction: {self.trend_direction} (1=up, -1=down)",
                f"  Protection Threshold: {self.config.trend_threshold * 100:.1f}%"
            ])
            
            if self.trend_protection_active:
                strategy = "Prioritizing SELLING to lock in profits" if self.trend_direction == 1 else "Cautious buying while reducing exposure"
                lines.extend([f"  Strategy: {strategy}"])
        
        # Candles and indicators section
        if self.candles is not None:
            candle_status = "Not Ready"
            if self.candles.ready:
                candle_status = "Ready"
                if not hasattr(self.candles, 'candles_df') or self.candles.candles_df is None or len(self.candles.candles_df) == 0:
                    candle_status = "Ready but no data"
            
            lines.extend([
                "\n  Technical Indicators",
                "  " + "=" * 70,
                f"  Candles: {self.config.candle_exchange} | Interval: {self.config.candle_interval} | Status: {candle_status}"
            ])
            
            if candle_status == "Ready" and hasattr(self.candles, 'candles_df') and self.candles.candles_df is not None and len(self.candles.candles_df) > 0:
                candles_df = self.get_candles_with_features()
                if candles_df is not None:
                    # Display latest indicator values
                    if f"NATR_{self.config.candle_length}" in candles_df.columns:
                        lines.extend([f"  Latest NATR: {candles_df[f'NATR_{self.config.candle_length}'].iloc[-1]:.6f}"])
                    
                    if f"RSI_{self.config.candle_length}" in candles_df.columns:
                        rsi = candles_df[f"RSI_{self.config.candle_length}"].iloc[-1]
                        rsi_condition = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                        lines.extend([f"  Latest RSI: {rsi:.2f} ({rsi_condition})"])
                    
                    # Add last few candles
                    lines.extend(["\n  Recent Candles:"])
                    try:
                        # Get only essential columns for display
                        display_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
                        available_cols = [col for col in display_cols if col in candles_df.columns]
                        
                        last_candles = candles_df[available_cols].tail(3).iloc[::-1]
                        if len(last_candles) > 0:
                            lines.extend(["    " + line for line in last_candles.to_string(index=False).split("\n")])
                    except Exception as e:
                        lines.append(f"    Error displaying candles: {str(e)}")
        else:
            lines.extend(["\n  Technical Indicators: Not initialized"])

        return "\n".join(lines)