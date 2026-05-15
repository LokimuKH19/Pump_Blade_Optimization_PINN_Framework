from __future__ import annotations

# Historical runnable examples kept out of SurrogateModeling.py.
# They are intentionally disabled helpers; the active formal workflow remains
# in the main module's __main__ block.

def _legacy_main_disabled() -> None:
    import numpy as np
    from pathlib import Path

    from SurrogateModeling import (
        SurrogateModeling,
        find_first_blade_params,
        make_pure_physics_debug_case,
        make_supervised_simulation_cases,
        seed_everything,
    )

    # 历史调试入口保留作参数参考；真正执行的主入口在文件末尾。
    seed_everything(42)
    simulation_folders = [
        Path("../BladeOptimizerLFR/CQ_20260514_115826_SIMULATION"),
    ]
    output_root = Path("surrogate_debug_outputs")
    save_dir = output_root / "simulation_supervised_cfno2d"
    save_dir.mkdir(parents=True, exist_ok=True)

    debug_n = 64
    debug_epochs = 5000
    checkpoint_path = save_dir / "surrogate_checkpoint.pt"

    print("\n========== Supervised simulation training ==========")
    train_cases = make_supervised_simulation_cases(
        simulation_folders,
        n=debug_n,
        mu=0.006,
        rho=10650.0,
        omega=-210.0 * 2.0 * np.pi / 60.0,
        qv=0.025,
        theta_sector_index=0,
    )
    trainer = SurrogateModeling(
        train_cases=train_cases,
        val_cases=train_cases,
        input_mode="both",
        batch_size=1,
        lr=1e-3,
        modes=8,
        high_modes=16,
        width=16,
        depth=4,
        z_padding=8,
        operator_variant="cfno",
        pressure_highpass_weight=1e8,
        pressure_highpass_normalized=True,
        data_weight=1.0,
        physics_weight=0.1,
        warmup_epochs=20,
        ramp_epochs=30,
        use_kkt_projection=False,
        kkt_projection_iters=24,
        kkt_projection_strength=0.35,
        learn_ibm_params=True,
        ibm_c_range=(0.3, 3.0),
        ibm_epsilon_range=(0.001, 0.05),
        micro_batch_size=None,
        slice_batch_size=None,
        auto_cuda_batching=True,
        cuda_memory_fraction=0.80,
        use_activation_checkpointing=True,
        device="cuda",
    )

    smoke = trainer.smoke_test(do_backward=True)
    print("Smoke test:", smoke)
    trainer.plot_blade_spans(case_index=0, spans=(0.4, 0.6), show=False, save_path=save_dir / "blade_spans.png")
    history = trainer.fit(epochs=debug_epochs, print_interval=1)
    trainer.plot_training_history(history, show=False, save_path=save_dir / "training_loss_log.png")
    trainer.save_checkpoint(checkpoint_path, history=history)
    trainer.post_process_case(
        case_index=0,
        spans=(0.4, 0.6),
        show=False,
        save_path=save_dir / "post_physical_spans.png",
        plot_3d=True,
        save_path_3d=save_dir / "post_3d_streamlines.png",
        passages_to_plot_3d=1,
    )
    trainer.plot_frequency_energy_trends(
        case_index=0,
        show=False,
        save_path=save_dir / "frequency_energy_trends.png",
        summary_path=save_dir / "frequency_energy_summary.json",
    )
    raise SystemExit(0)

    # 这里保留一个最直接的纯物理调试入口：
    # 自动寻找 blade_params.json，找到就跑 smoke test 和纯物理训练。
    seed_everything(42)
    blade_params = find_first_blade_params("../BladeOptimizerLFR/CQ_20260514_115826_SIMULATION")
    if blade_params is not None:
        output_root = Path("surrogate_debug_outputs")
        pure_physics_runs = (   # 在这里预约运行的内容
            ("fno2d", "fno", output_root / "cfno2d_pure_physics"),
            #("wno3d", "wno3d", output_root / "wno3d_pure_physics"),
        )
        debug_n = 64
        debug_epochs = 5000

        for run_name, operator_variant, save_dir in pure_physics_runs:
            print(f"\n========== Pure physics debug: {run_name} ({operator_variant}) ==========")
            checkpoint_path = save_dir / "surrogate_checkpoint.pt"
            trainer = SurrogateModeling.build_pure_physics_debug_trainer(
                blade_params=blade_params,
                n=debug_n,
                # micro_batch_size=None,  # 自动按 CUDA 显存估算
                # slice_batch_size=None,  # 2D 算子自动切 R-slices
                # auto_cuda_batching=True,
                # cuda_memory_fraction=0.80,
                # 物理定义
                mu=0.006,
                rho=10650.0,
                omega=-210.0 * 2 * np.pi / 60.0,
                qv=0.025,
                lr=1e-3,
                learn_ibm_params=True,
                ibm_c_range=(0.3, 3.0),
                operator_variant=operator_variant,
                modes=8,
                high_modes=16,
                width=16,
                depth=4,
                pressure_highpass_weight=1e8,
                pressure_highpass_normalized=True,
                use_kkt_projection=False,
                kkt_projection_iters=24,
                kkt_projection_strength=0.35,
                ibm_epsilon_range=(0.001, 0.05),
            )

            smoke = trainer.smoke_test(do_backward=True)
            print(f"Smoke test ({run_name}):", smoke)
            trainer.fit_pure_physics_debug(
                epochs=debug_epochs,
                print_interval=1,
                preview_spans=(0.4, 0.6),
                post_spans=(0.4, 0.6),
                show_plots=False,
                save_dir=save_dir,
                save_checkpoint_path=checkpoint_path,
                plot_3d=True,
            )

            fine_case = make_pure_physics_debug_case(
                blade_params=blade_params,
                n=256,
                mu=0.0016,
                rho=10650.0,
                omega=-210.0 * 2 * np.pi / 60.0,
                qv=0.025,
            )
            SurrogateModeling.deploy_from_checkpoint(
                checkpoint_path,
                fine_case,
                show=False,
                save_path_2d=save_dir / "fine_grid_spans.png",
                save_path_3d=save_dir / "fine_grid_3d_streamlines.png",
                spans=(0.4, 0.6),
                plot_3d=True,
            )
    else:
        print("No blade_params.json found. Use build_pure_physics_debug_trainer(...) or load_cases_from_pt(...).")


def _previous_formal_main_disabled() -> None:
    import json
    import numpy as np
    from pathlib import Path

    from SurrogateModeling import (
        FlowCaseConfig,
        SurrogateModeling,
        case_summary,
        make_pure_physics_debug_case,
        make_pure_physics_debug_cases,
        make_supervised_simulation_cases,
        seed_everything,
    )

    # 上一版入口保留作参数参考；实际运行入口在文件末尾。
    # ============================================================
    # 0. 运行入口
    # ============================================================
    # TRAINING_MODE 控制同一套主程序的三种用法：
    # mixed      : 先用 FLUENT 数据把流场锚住，再逐步加入物理残差，推荐作为正式训练入口。
    # data_only  : 只看 CSV 插值和监督学习是否健康，适合排查数据导入/归一化。
    # physics_only: 不使用 CSV 标签，只用 blade_params.json 做纯物理调试。
    TRAINING_MODE = "mixed"
    seed_everything(42)

    simulation_folders = [
        Path("../BladeOptimizerLFR/CQ_20260514_115826_SIMULATION"),    # 确保这个文件夹拥有一个装着json和一个装着流场的csv
        # 后续增加叶型时，在这里继续追加类似文件夹即可：
        # Path("../BladeOptimizerLFR/CQ_xxxxx_SIMULATION"),
    ]

    output_root = Path("surrogate_debug_outputs")
    run_name = f"simulation_{TRAINING_MODE}_cfno2d"
    save_dir = output_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "surrogate_checkpoint.pt"

    # ============================================================
    # 1. 物理定义
    # ============================================================
    # 这里集中放“真实物理量”，避免训练配置和数据导入配置里重复写一遍。
    # rpm 保留工程语义，omega 是求解器实际使用的角速度(rad/s)。
    rpm = -210.0
    physics_config = {
        "mu": 0.006,
        "rho": 10650.0,
        "omega": float(rpm * 2.0 * np.pi / 60.0),
        "qv": 0.025,
        "g": 9.8,
    }

    # ============================================================
    # 2. 数据与网格定义
    # ============================================================
    # n 是代理模型训练网格数。CSV 中的点是非均匀散点，会先截取 0 到 360/n_blade 度
    # 的单流道，再投射到这个无量纲 R-Theta-Z 网格上。
    data_config = {
        "n": 64,
        "theta_sector_index": 0,
        "interpolation_chunk_size": 2.5e5,
    }

    # ============================================================
    # 3. 算子网络定义
    # ============================================================
    # operator_variant 可选：cfno / fno / hf_cfno / wno2d / wno3d。
    # 当前默认使用 2D CFNO，并沿 R 方向共享权重；网格很细时 slice_batch_size 会自动估算，从而采用对应的策略求解。
    model_config = {
        "operator_variant": "cfno",
        "input_mode": "both",
        "modes": 8,
        "high_modes": 16,
        "width": 16,
        "depth": 4,
        "z_padding": 8,
        "fourier_feature_bands": (1, 2, 4, 8),
        "hf_high_gate_init": -1.0,
        "hf_use_local_highpass": True,
        "pressure_smoothing": 0.0,
        # pressure_supervision_mode:
        # value    : 回退到直接用 P 通道做数据监督。
        # gradient : 只用压力梯度做数据监督，适合压力零点不可信的 CSV。
        # none     : 数据损失只监督速度。
        "pressure_supervision_mode": "value",
        # origin 会在网络输出端强制 P(0,0,0)=0；混合训练会在下面自动改成 none。
        "pressure_reference_mode": "origin",
        # value 模式下用 absolute 才能学习并绘制 Fluent 原始 P；gradient 模式可改回 training_origin。
        "pressure_data_reference": "absolute",
    }

    # ============================================================
    # 4. 物理-数据协同训练策略
    # ============================================================
    # 推荐工作流：
    # A. data warmup：前 warmup_epochs 只拟合 FLUENT 数据，先学到正确量级和主流动结构。
    # B. physics ramp：接下来 ramp_epochs 逐步增加物理残差，避免一开始被离散残差牵偏。
    # C. joint training：后续数据损失负责贴近仿真，连续方程/动量方程/流量约束负责补齐未采样区域。
    training_config = {
        "epochs": 5000,
        "print_interval": 1,
        "batch_size": 1,
        "lr": 1e-3,
        "pressure_highpass_weight": 1e8,
        "pressure_highpass_normalized": True,
        "use_kkt_projection": False,
        "kkt_projection_iters": 24,
        "kkt_projection_strength": 0.35,
        "learn_ibm_params": True,
        "ibm_c_range": (0.3, 3.0),
        "ibm_epsilon_range": (0.001, 0.05),
        "micro_batch_size": None,
        "slice_batch_size": None,
        "auto_cuda_batching": True,
        "cuda_memory_fraction": 0.80,
        "use_activation_checkpointing": True,
        "device": "cuda",
    }
    mode_presets = {
        "mixed": {
            "data_weight": 1.0,
            "physics_weight": 0.1,
            "warmup_epochs": 20,
            "ramp_epochs": 30,
        },
        "data_only": {
            "data_weight": 1.0,
            "physics_weight": 0.0,
            "warmup_epochs": 0,
            "ramp_epochs": 0,
        },
        "physics_only": {
            "data_weight": 0.0,
            "physics_weight": 1.0,
            "warmup_epochs": 0,
            "ramp_epochs": 0,
        },
    }
    if TRAINING_MODE not in mode_presets:
        raise ValueError(f"Unknown TRAINING_MODE={TRAINING_MODE!r}; choose from {sorted(mode_presets)}.")
    training_config.update(mode_presets[TRAINING_MODE])

    print(f"\n========== Surrogate workflow: {TRAINING_MODE} ==========")
    print(f"输出目录: {save_dir}")
    print(
        "物理定义: "
        f"rpm={rpm:.6g}, omega={physics_config['omega']:.6g}, "
        f"mu={physics_config['mu']:.6g}, rho={physics_config['rho']:.6g}, "
        f"qv={physics_config['qv']:.6g}, g={physics_config['g']:.6g}"
    )

    # ============================================================
    # 5. 构造训练样本
    # ============================================================
    if TRAINING_MODE == "physics_only":
        # 纯物理模式只依赖几何文件。它适合检查边界、流量约束、IBM 参数是否能打通。
        blade_param_files = [folder / "blade_params.json" for folder in simulation_folders]
        train_cases = make_pure_physics_debug_cases(
            blade_params=blade_param_files,
            n=data_config["n"],
            **physics_config,
        )
    else:
        # 数据/混合模式会在每个仿真文件夹中自动寻找唯一 CSV，并与 blade_params.json 配对。
        train_cases = make_supervised_simulation_cases(
            simulation_folders,
            n=data_config["n"],
            theta_sector_index=data_config["theta_sector_index"],
            interpolation_chunk_size=data_config["interpolation_chunk_size"],
            **physics_config,
        )
    val_cases = train_cases

    first_config = FlowCaseConfig.from_mapping(train_cases[0])
    run_summary = {
        "training_mode": TRAINING_MODE,
        "simulation_folders": [str(folder) for folder in simulation_folders],
        "physics_config": {"rpm": rpm, **physics_config, "g_star": first_config.g_star, "P0": first_config.P0},
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config,
        "case_summaries": [case_summary(case) for case in train_cases],
    }
    summary_path = save_dir / "run_config_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"运行配置摘要已保存到: {summary_path}")

    # ============================================================
    # 6. 建立训练器
    # ============================================================
    trainer = SurrogateModeling(
        train_cases=train_cases,
        val_cases=val_cases,
        input_mode=model_config["input_mode"],
        batch_size=training_config["batch_size"],
        lr=training_config["lr"],
        modes=model_config["modes"],
        high_modes=model_config["high_modes"],
        width=model_config["width"],
        depth=model_config["depth"],
        z_padding=model_config["z_padding"],
        operator_variant=model_config["operator_variant"],
        fourier_feature_bands=model_config["fourier_feature_bands"],
        hf_high_gate_init=model_config["hf_high_gate_init"],
        hf_use_local_highpass=model_config["hf_use_local_highpass"],
        pressure_smoothing=model_config["pressure_smoothing"],
        pressure_highpass_weight=training_config["pressure_highpass_weight"],
        pressure_highpass_normalized=training_config["pressure_highpass_normalized"],
        data_weight=training_config["data_weight"],
        physics_weight=training_config["physics_weight"],
        warmup_epochs=training_config["warmup_epochs"],
        ramp_epochs=training_config["ramp_epochs"],
        use_kkt_projection=training_config["use_kkt_projection"],
        kkt_projection_iters=training_config["kkt_projection_iters"],
        kkt_projection_strength=training_config["kkt_projection_strength"],
        learn_ibm_params=training_config["learn_ibm_params"],
        ibm_c_range=training_config["ibm_c_range"],
        ibm_epsilon_range=training_config["ibm_epsilon_range"],
        micro_batch_size=training_config["micro_batch_size"],
        slice_batch_size=training_config["slice_batch_size"],
        auto_cuda_batching=training_config["auto_cuda_batching"],
        cuda_memory_fraction=training_config["cuda_memory_fraction"],
        use_activation_checkpointing=training_config["use_activation_checkpointing"],
        device=training_config["device"],
    )

    # ============================================================
    # 7. 训练前检查、训练、保存结果
    # ============================================================
    # smoke_test 会打通前向、损失和 backward；如果 CUDA 容量不足，训练器会自动尝试更小 batch。
    smoke = trainer.smoke_test(do_backward=True)
    print("Smoke test:", smoke)

    preview_spans = (0.4, 0.6)
    trainer.plot_blade_spans(
        case_index=0,
        spans=preview_spans,
        show=False,
        save_path=save_dir / "blade_spans.png",
    )
    history = trainer.fit(
        epochs=training_config["epochs"],
        print_interval=training_config["print_interval"],
    )
    trainer.plot_training_history(
        history,
        show=False,
        save_path=save_dir / "training_loss_log.png",
    )
    trainer.save_checkpoint(
        checkpoint_path,
        history=history,
        extra_metadata={"run_summary_path": str(summary_path), "training_mode": TRAINING_MODE},
    )

    # ============================================================
    # 8. 后处理诊断
    # ============================================================
    # 2D span 图用于看数值场，3D streamline 现在默认白底并按流线种子使用高对比配色。
    trainer.post_process_case(
        case_index=0,
        spans=preview_spans,
        show=False,
        save_path=save_dir / "post_physical_spans.png",
        plot_3d=True,
        save_path_3d=save_dir / "post_3d_streamlines.png",
        passages_to_plot_3d=1,
    )
    trainer.plot_frequency_energy_trends(
        case_index=0,
        show=False,
        save_path=save_dir / "frequency_energy_trends.png",
        summary_path=save_dir / "frequency_energy_summary.json",
    )

    # ============================================================
    # 9. 可选：细网格部署检查
    # ============================================================
    # 训练完成后若需要测试“粗网格训练 -> 细网格推理”，改成 True 即可。
    # 这里用纯几何 case 做部署，不再依赖 CSV 标签，因此可以检查模型泛化出的物理场。
    run_fine_grid_deploy = False
    if run_fine_grid_deploy:
        fine_case = make_pure_physics_debug_case(
            blade_params=simulation_folders[0] / "blade_params.json",
            n=256,
            **physics_config,
        )
        SurrogateModeling.deploy_from_checkpoint(
            checkpoint_path,
            fine_case,
            show=False,
            save_path_2d=save_dir / "fine_grid_spans.png",
            save_path_3d=save_dir / "fine_grid_3d_streamlines.png",
            spans=preview_spans,
            plot_3d=True,
        )
