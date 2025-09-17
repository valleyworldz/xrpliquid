class HyperliquidVaultManager:
    """🏦 HYPERLIQUID VAULT MANAGEMENT SYSTEM"""
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
        self.vault_config = {
            'hlp_vault_enabled': True,
            'auto_compound': True,
            'target_allocation_pct': 20.0,  # 20% of portfolio in HLP
            'min_vault_deposit': 100.0,     # $100 minimum
            'compound_threshold': 50.0,     # Compound when $50+ rewards
        }
    def check_hlp_vault_status(self):
        try:
            vault_info = {
                'is_participating': False,
                'hlp_balance': 0.0,
                'pending_rewards': 0.0,
                'apy_estimate': 10.5,
                'vault_tvl': 0.0,
                'user_share_pct': 0.0
            }
            try:
                user_state = self.api.get_user_state()
                if user_state and 'vaultEquity' in user_state:
                    vault_equity = safe_float(user_state.get('vaultEquity', 0))
                    if vault_equity > 0:
                        vault_info['is_participating'] = True
                        vault_info['hlp_balance'] = vault_equity
                        daily_rate = vault_info['apy_estimate'] / 365 / 100
                        vault_info['estimated_daily_rewards'] = vault_equity * daily_rate
                        self.logger.info(f"🏦 HLP Vault: ${vault_equity:.2f} balance, ~${vault_info['estimated_daily_rewards']:.2f}/day")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not fetch vault data: {e}")
            return vault_info
        except Exception as e:
            self.logger.error(f"❌ HLP vault status check failed: {e}")
            return {'is_participating': False, 'hlp_balance': 0.0}
    def optimize_hlp_allocation(self, account_value):
        try:
            vault_status = self.check_hlp_vault_status()
            target_allocation = account_value * (self.vault_config['target_allocation_pct'] / 100)
            current_allocation = vault_status.get('hlp_balance', 0.0)
            allocation_diff = target_allocation - current_allocation
            recommendation = {
                'action': 'hold',
                'amount': 0.0,
                'reason': 'Allocation optimal',
                'current_allocation': current_allocation,
                'target_allocation': target_allocation,
                'allocation_pct': (current_allocation / account_value * 100) if account_value > 0 else 0
            }
            if abs(allocation_diff) < self.vault_config['min_vault_deposit']:
                recommendation['reason'] = f"Allocation within target range (${allocation_diff:+.2f})"
            elif allocation_diff > self.vault_config['min_vault_deposit']:
                recommendation['action'] = 'deposit'
                recommendation['amount'] = allocation_diff
                recommendation['reason'] = f"Deposit ${allocation_diff:.2f} to reach {self.vault_config['target_allocation_pct']}% target"
            elif allocation_diff < -self.vault_config['min_vault_deposit']:
                recommendation['action'] = 'withdraw'
                recommendation['amount'] = abs(allocation_diff)
                recommendation['reason'] = f"Withdraw ${abs(allocation_diff):.2f} to reach {self.vault_config['target_allocation_pct']}% target"
            self.logger.info(f"🏦 HLP Optimization: {recommendation['reason']}")
            return recommendation
        except Exception as e:
            self.logger.error(f"❌ HLP allocation optimization failed: {e}")
            return {'action': 'hold', 'amount': 0.0, 'reason': 'Error in optimization'}
    def execute_vault_action(self, action, amount):
        try:
            if action == 'deposit' and amount >= self.vault_config['min_vault_deposit']:
                self.logger.info(f"🏦 Executing HLP deposit: ${amount:.2f}")
                return True
            elif action == 'withdraw' and amount >= self.vault_config['min_vault_deposit']:
                self.logger.info(f"🏦 Executing HLP withdrawal: ${amount:.2f}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Vault action execution failed: {e}")
            return False

class HypeStakingManager:
    """💎 HYPE STAKING FEE DISCOUNT SYSTEM"""
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
        self.staking_config = {
            'auto_stake_enabled': True,
            'target_discount_tier': 3,
            'min_stake_amount': 50.0,
            'stake_threshold': 100.0,
        }
        self.discount_tiers = {
            1: {'hype_required': 1000, 'fee_discount': 0.10},
            2: {'hype_required': 5000, 'fee_discount': 0.20},
            3: {'hype_required': 15000, 'fee_discount': 0.30},
            4: {'hype_required': 50000, 'fee_discount': 0.40},
        }
    def check_hype_staking_status(self):
        try:
            staking_status = {
                'hype_balance': 0.0,
                'staked_hype': 0.0,
                'current_tier': 0,
                'fee_discount': 0.0,
                'estimated_savings': 0.0,
                'next_tier_required': 0.0
            }
            try:
                user_state = self.api.get_user_state()
                staked_amount = staking_status['staked_hype']
                for tier, requirements in self.discount_tiers.items():
                    if staked_amount >= requirements['hype_required']:
                        staking_status['current_tier'] = tier
                        staking_status['fee_discount'] = requirements['fee_discount']
                next_tier = staking_status['current_tier'] + 1
                if next_tier in self.discount_tiers:
                    staking_status['next_tier_required'] = (
                        self.discount_tiers[next_tier]['hype_required'] - staked_amount
                    )
                self.logger.info(f"💎 HYPE Staking: Tier {staking_status['current_tier']} "
                               f"({staking_status['fee_discount']*100:.0f}% discount)")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not fetch HYPE staking data: {e}")
            return staking_status
        except Exception as e:
            self.logger.error(f"❌ HYPE staking status check failed: {e}")
            return {'current_tier': 0, 'fee_discount': 0.0}
    def optimize_hype_staking(self):
        try:
            staking_status = self.check_hype_staking_status()
            optimization = {
                'action': 'hold',
                'stake_amount': 0.0,
                'potential_savings': 0.0,
                'roi_estimate': 0.0,
                'reason': 'Staking optimal'
            }
            current_tier = staking_status['current_tier']
            target_tier = self.staking_config['target_discount_tier']
            if current_tier < target_tier:
                next_tier_hype_needed = staking_status.get('next_tier_required', 0)
                if next_tier_hype_needed > 0:
                    current_discount = staking_status['fee_discount']
                    target_discount = self.discount_tiers[target_tier]['fee_discount']
                    estimated_monthly_volume = 50000.0
                    estimated_fees = estimated_monthly_volume * 0.0003
                    additional_savings = estimated_fees * (target_discount - current_discount)
                    optimization = {
                        'action': 'stake',
                        'stake_amount': next_tier_hype_needed,
                        'potential_savings': additional_savings,
                        'roi_estimate': additional_savings / next_tier_hype_needed if next_tier_hype_needed > 0 else 0,
                        'reason': f'Stake {next_tier_hype_needed} HYPE to reach tier {target_tier}'
                    }
            self.logger.info(f"💎 HYPE Staking Optimization: {optimization['reason']}")
            return optimization
        except Exception as e:
            self.logger.error(f"❌ HYPE staking optimization failed: {e}")
            return {'action': 'hold', 'stake_amount': 0.0, 'reason': 'Error in optimization'} 