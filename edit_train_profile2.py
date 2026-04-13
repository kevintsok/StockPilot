filepath = '/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/train.py'
with open(filepath, 'r') as f:
    content = f.read()

old_block = '''# Profile backward pass only on step 3 to avoid CUDA OOM.
            # Forward runs without profiler instrumentation to save GPU memory.
            profile_backward_only = profile_enabled and batch_idx == 3

            # ── Forward pass (no profiler) ─────────────────────────────────────
            optimizer.zero_grad()
            out = model(x)
            if len(out) == 4:
                pred_reg, pred_cls, pred_reg_all, pred_cls_all = out
                total_reg_loss = 0.0
                total_cls_loss = 0.0
                total_rank_loss = 0.0
                for h_idx, h in enumerate(cfg.horizons):
                    y_reg_h = y_reg_or_dict[h].to(device)
                    pred_reg_h = pred_reg_all[h_idx]
                    min_len = min(pred_reg_h.shape[-1], y_reg_h.shape[-1])
                    total_reg_loss += loss_fn_reg(pred_reg_h[:, :min_len], y_reg_h[:, :min_len])
                    pred_cls_h = pred_cls_all[h_idx]
                    total_cls_loss += loss_fn_cls(pred_cls_h[:, :min_len], y_cls[:, :min_len])
                    total_rank_loss += _compute_ranking_loss(pred_reg_h[:, -1:], y_reg_h[:, -1:])
                reg_loss = total_reg_loss / len(cfg.horizons)
                cls_loss = total_cls_loss / len(cfg.horizons)
                rank_loss = total_rank_loss / len(cfg.horizons)
            else:
                y_reg = y_reg_or_dict.to(device)
                pred_reg, pred_cls = out
                reg_loss = loss_fn_reg(pred_reg, y_reg)
                cls_loss = loss_fn_cls(pred_cls, y_cls)
                rank_loss = _compute_ranking_loss(pred_reg, y_reg)
            lambda_reg = getattr(cfg, "lambda_reg", 0.1)
            lambda_cls = getattr(cfg, "lambda_cls", 10.0)
            lambda_rank = getattr(cfg, "lambda_rank", 1.0)
            loss = lambda_reg * reg_loss + lambda_cls * cls_loss + lambda_rank * rank_loss

            # ── Backward pass (profiled) ───────────────────────────────────────
            if profile_backward_only:
                try:
                    profile_trace_path.parent.mkdir(parents=True, exist_ok=True)
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(REPORT_DIR)),
                        record_shapes=True,
                        with_stack=True,
                        with_modules=True,
                    ) as prof:
                        loss.backward()
                        if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                            clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                        optimizer.step()
                        scheduler.step()
                    print(f"[Profile] trace exported to {REPORT_DIR}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[Profile] error: {exc}")
                    loss.backward()
                    if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                        clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
            else:
                loss.backward()
                if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item()'''

new_block = '''# Profile backward pass on steps 3-5 to avoid CUDA OOM.
            # Forward runs without profiler instrumentation to save GPU memory.
            # Uses schedule(wait=0, active=3) so the same profiler instance
            # records all three steps into one trace file.
            profile_this_step = profile_enabled and 3 <= batch_idx <= 5
            if profile_this_step and _active_profiler is None:
                profile_trace_path.parent.mkdir(parents=True, exist_ok=True)
                _active_profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(REPORT_DIR)),
                    record_shapes=True,
                    with_stack=True,
                    with_modules=True,
                    schedule=torch.profiler.schedule(wait=0, active=3, repeat=1),
                )
                _active_profiler.__enter__()

            # ── Forward pass (no profiler) ─────────────────────────────────────
            optimizer.zero_grad()
            out = model(x)
            if len(out) == 4:
                pred_reg, pred_cls, pred_reg_all, pred_cls_all = out
                total_reg_loss = 0.0
                total_cls_loss = 0.0
                total_rank_loss = 0.0
                for h_idx, h in enumerate(cfg.horizons):
                    y_reg_h = y_reg_or_dict[h].to(device)
                    pred_reg_h = pred_reg_all[h_idx]
                    min_len = min(pred_reg_h.shape[-1], y_reg_h.shape[-1])
                    total_reg_loss += loss_fn_reg(pred_reg_h[:, :min_len], y_reg_h[:, :min_len])
                    pred_cls_h = pred_cls_all[h_idx]
                    total_cls_loss += loss_fn_cls(pred_cls_h[:, :min_len], y_cls[:, :min_len])
                    total_rank_loss += _compute_ranking_loss(pred_reg_h[:, -1:], y_reg_h[:, -1:])
                reg_loss = total_reg_loss / len(cfg.horizons)
                cls_loss = total_cls_loss / len(cfg.horizons)
                rank_loss = total_rank_loss / len(cfg.horizons)
            else:
                y_reg = y_reg_or_dict.to(device)
                pred_reg, pred_cls = out
                reg_loss = loss_fn_reg(pred_reg, y_reg)
                cls_loss = loss_fn_cls(pred_cls, y_cls)
                rank_loss = _compute_ranking_loss(pred_reg, y_reg)
            lambda_reg = getattr(cfg, "lambda_reg", 0.1)
            lambda_cls = getattr(cfg, "lambda_cls", 10.0)
            lambda_rank = getattr(cfg, "lambda_rank", 1.0)
            loss = lambda_reg * reg_loss + lambda_cls * cls_loss + lambda_rank * rank_loss

            # ── Backward pass (profiled) ─────────────────────────────────────
            if profile_this_step:
                _active_profiler.step()
                loss.backward()
                if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                if batch_idx == 5:   # last profiled step — export and stop
                    try:
                        _active_profiler.__exit__(None, None, None)
                        print(f"[Profile] trace exported to {REPORT_DIR}")
                    except Exception as exc:
                        print(f"[Profile] export error: {exc}")
                    _active_profiler = None
            else:
                loss.backward()
                if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item()'''

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Block replaced successfully")
else:
    print("ERROR: Block not found!")
    # Debug
    idx = content.find('profile_backward_only')
    print(f"profile_backward_only at index: {idx}")
    # Check if the block exists partially
    if '# Profile backward pass only on step 3' in content:
        print("Found start marker")
    if 'running_loss += loss.item()' in content:
        print("Found end marker")

with open(filepath, 'w') as f:
    f.write(content)
print("File written")
