#!/usr/bin/env python3
with open('src/auto_select_stock/predict/strategies/__init__.py', 'r') as f:
    lines = f.readlines()

new_select = '''    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        # orig_ep: FIXED original entry price (never changes after purchase)
        orig_ep_cache = cache.setdefault("_conf_orig_ep", {})
        # curr_px: current price = previous day's close (updated each day via realized_rets)
        curr_px_cache = cache.setdefault("_conf_curr_px", {})
        held_cache = cache.setdefault("_conf_held", set())

        # ── Stop-loss / take-profit on previously held positions ─────
        # Compare CURRENT price (yesterday's close) vs ORIGINAL entry price (fixed)
        if self.stop_loss_pct > 0 or self.take_profit_pct > 0:
            for sym in list(held_cache):
                orig_ep = orig_ep_cache.get(sym)
                if orig_ep is None:
                    held_cache.discard(sym)
                    continue
                curr_px = curr_px_cache.get(sym, orig_ep)
                pct = (curr_px / orig_ep) - 1.0 if orig_ep > 0 else 0.0
                if pct < -self.stop_loss_pct or pct > self.take_profit_pct:
                    held_cache.discard(sym)
                    orig_ep_cache.pop(sym, None)
                    curr_px_cache.pop(sym, None)

        # ── Determine candidate signals ───────────────────────────────
        if self.allow_short:
            long_sigs = [s for s in signals if self._get_predicted_ret(s) > self.min_confidence]
            short_sigs = [s for s in signals if self._get_predicted_ret(s) < -self.min_confidence]
            top_long = sorted(long_sigs, key=lambda s: self._get_predicted_ret(s), reverse=True)[: self.top_k]
            top_short = sorted(short_sigs, key=lambda s: self._get_predicted_ret(s))[: self.top_k]
            all_sigs = top_long + top_short
        else:
            candidates = [s for s in signals if self._get_predicted_ret(s) > self.min_confidence]
            all_sigs = sorted(candidates, key=lambda s: self._get_predicted_ret(s), reverse=True)[: self.top_k]

        if not all_sigs:
            return {}

        # ── Record original entry price for newly selected positions ───
        for s in all_sigs:
            if s.symbol not in held_cache:
                orig_ep_cache[s.symbol] = s.entry_price
                curr_px_cache[s.symbol] = s.entry_price

        held_cache.update(s.symbol for s in all_sigs)

        # ── Compute confidence weights ───────────────────────────────
        total_conf = sum(abs(self._get_predicted_ret(s)) for s in all_sigs)
        if total_conf <= 0:
            return {}

        weights = {}
        long_alloc = 1.0 if not self.allow_short else 0.5
        short_alloc = 0.0 if not self.allow_short else -0.5

        for s in all_sigs:
            conf = abs(self._get_predicted_ret(s)) / total_conf
            if self._get_predicted_ret(s) > 0:
                weights[s.symbol] = long_alloc * conf
            else:
                weights[s.symbol] = short_alloc * conf
        return weights

'''

new_on_day = '''    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        orig_ep_cache = cache.setdefault("_conf_orig_ep", {})
        curr_px_cache = cache.setdefault("_conf_curr_px", {})
        held_cache = cache.setdefault("_conf_held", set())

        portfolio = cache.get("_portfolio", {})
        for sym in list(held_cache):
            pos = portfolio.get(sym)
            if pos is not None:
                orig_ep_cache[sym] = pos.get("entry_price", orig_ep_cache.get(sym, 0.0))

        for sym in list(held_cache):
            ret = realized_rets.get(sym)
            if ret is None:
                held_cache.discard(sym)
                orig_ep_cache.pop(sym, None)
                curr_px_cache.pop(sym, None)
                continue
            orig_ep = orig_ep_cache.get(sym, 0.0)
            if orig_ep <= 0:
                held_cache.discard(sym)
                orig_ep_cache.pop(sym, None)
                curr_px_cache.pop(sym, None)
                continue
            # Update current price (yesterday's close) = orig_ep * (1 + realized_ret)
            new_curr = orig_ep * (1.0 + ret)
            curr_px_cache[sym] = new_curr
            # orig_ep stays FIXED — always compare against original purchase price


'''

# Find ConfStopStrategy line (next class after ConfidenceStrategy)
stop_line = None
for i, line in enumerate(lines):
    if 'class ConfidenceStopStrategy' in line:
        stop_line = i
        break

print(f"ConfidenceStrategy ends before line {stop_line+1}")
print(f"select_positions at 408, on_day_end at 479 (1-indexed)")

# Lines 407-476 (0-indexed) = select_positions body (ends before on_day_end at 478)
# on_day_end at 478-516 (0-indexed)
# ConfStopStrategy at 520 (0-indexed)

new_lines = lines[:407]  # before select_positions (line 408, 0-indexed 407)
new_lines.append(new_select)
new_lines.extend(lines[477:478])  # blank lines before on_day_end
new_lines.append(new_on_day)
new_lines.extend(lines[520:])  # ConfStopStrategy onwards

with open('src/auto_select_stock/predict/strategies/__init__.py', 'w') as f:
    f.writelines(new_lines)

print(f"Done! Original: {len(lines)} lines, New: {len(new_lines)} lines")
