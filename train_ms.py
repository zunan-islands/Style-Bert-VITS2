from __future__ import annotations

import argparse
import datetime
import gc
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

from style_bert_vits2.constants import (
    DEFAULT_TRAIN_ENV,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from style_bert_vits2.nlp.symbols import SYMBOLS
from style_bert_vits2.utils.paths import (
    TrainingModelPaths,
    add_dataset_root_argument,
    add_model_argument,
    get_paths_config,
)
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT

# logging.getLogger("numba").setLevel(logging.WARNING)
from training import default_style
from training.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from training.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from training.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from training.runtime import TrainRuntimeConfig
from training.utils import check_git_hash, is_resuming, summarize


if TYPE_CHECKING:
    from loguru import Logger as LoguruLogger


# PyTorch 最適化設定 (torch >= 2.1 前提)
# TF32: Ampere 以降の GPU で FP32 演算を高速化
torch.backends.cuda.matmul.allow_tf32 = True
# If encountered training problem, please try to disable TF32.
torch.backends.cudnn.allow_tf32 = True
# 行列演算精度: "medium" は TF32 を使用し、速度と精度のバランスを取る
torch.set_float32_matmul_precision("medium")
# cuDNN ベンチマーク: 固定入力サイズの学習で畳み込みアルゴリズムを自動選択し高速化
torch.backends.cudnn.benchmark = True
# Scaled Dot-Product Attention (SDPA) バックエンド設定
# Flash Attention: 最も高速だがハードウェア要件あり (Ampere 以降)
torch.backends.cuda.enable_flash_sdp(True)
# Memory Efficient Attention: Flash が使えない場合のフォールバック
torch.backends.cuda.enable_mem_efficient_sdp(True)
# Math Attention: 上記が使えない場合の最終フォールバック
torch.backends.cuda.enable_math_sdp(True)

global_step = 0
# エポックの途中から再開したとき、そのエポックで学習済みの先頭のバッチ数 (最初のエポックだけ飛ばす)
resume_skip_batches = 0

api = HfApi()


def run():
    paths_config = get_paths_config()
    parser = argparse.ArgumentParser(
        description="Train multi-language model.",
    )
    add_model_argument(parser)
    add_dataset_root_argument(parser)
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default=None,
        help="Directory that contains G_0.safetensors / D_0.safetensors for initialization. "
        "If omitted, model_dir is used.",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        help="Root directory of model assets needed for inference.",
        default=str(paths_config.assets_root),
    )
    parser.add_argument(
        "--keep_ckpts",
        type=int,
        default=1,
        help="Number of checkpoints to keep. Set to 0 to keep all. Default: 1",
    )
    parser.add_argument(
        "--skip_default_style",
        action="store_true",
        help="Skip saving default style config and mean vector.",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Do not show the progress bar while training.",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Speed up training by disabling logging and evaluation.",
    )
    parser.add_argument(
        "--repo_id",
        help="Huggingface model repo id to backup the model.",
        default=None,
    )
    parser.add_argument(
        "--not_use_custom_batch_sampler",
        help="Don't use custom batch sampler for training, which was used in the version < 2.5",
        action="store_true",
    )
    args = parser.parse_args()

    # TrainingModelPaths を使ってパスを解決
    model_folder_name: str = args.model
    paths = TrainingModelPaths(
        model_folder_name=model_folder_name, dataset_root=Path(args.dataset_root)
    )
    assets_root = Path(args.assets_root)

    # チェックポイント保存ディレクトリ (Data/{model_folder_name}/models/)
    model_dir = str(paths.models_dir)
    pretrained_model_dir = args.pretrained_model_dir or model_dir
    if args.pretrained_model_dir is not None and not os.path.isdir(
        pretrained_model_dir
    ):
        logger.warning(
            f"Pretrained model dir not found: {pretrained_model_dir}. Falling back to model_dir."
        )
        pretrained_model_dir = model_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(str(paths.dataset_dir / f"train_{timestamp}.log"))

    # 分散学習用の環境変数をセット
    # 未設定の変数のみ DEFAULT_TRAIN_ENV から設定される
    for env_name, env_value in DEFAULT_TRAIN_ENV.items():
        if env_name not in os.environ.keys():
            logger.info(f"Setting default environment variable: {env_name}={env_value}")
            os.environ[env_name] = str(env_value)
    logger.info(
        "Loading environment variables \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()

    hps = HyperParameters.load_from_json(paths.config_path)
    runtime_config = TrainRuntimeConfig(
        model_name=hps.model_name,
        model_dir=model_dir,
        out_dir=str(assets_root / model_folder_name),
        dataset_path=str(paths.dataset_dir),
        keep_ckpts=args.keep_ckpts,
        repo_id=args.repo_id,
        speedup=args.speedup,
        spec_cache=True,
    )

    """
    パス定数について:
    - args.model: 学習データのフォルダ名 (Data/ 以下)
    - runtime_config.model_name: config.json の model_name フィールド（ファイル名に使用）
    - runtime_config.model_dir: チェックポイント保存先 (Data/{model}/models/)
    - runtime_config.out_dir: 推論用モデルの出力先 (model_assets/{model}/)
    - runtime_config.dataset_path: データセットパス (Data/{model}/)
    """

    if args.repo_id is not None:
        # First try to upload config.json to check if the repo exists
        assert runtime_config.dataset_path is not None
        try:
            api.upload_file(
                path_or_fileobj=str(paths.config_path),
                path_in_repo=f"Data/{runtime_config.model_name}/config.json",
                repo_id=args.repo_id,
            )
        except Exception as ex:
            logger.error(
                f"Failed to upload files to the repo {args.repo_id}. "
                "Please check if the repo exists and you have logged in using `huggingface-cli login`.",
                exc_info=ex,
            )
            raise ex
        # Upload Data dir for resuming training
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=runtime_config.dataset_path,
            path_in_repo=f"Data/{runtime_config.model_name}",
            delete_patterns="*.pth",  # Only keep the latest checkpoint
            ignore_patterns="raw/**",  # Ignore raw data
            run_as_future=True,
        )

    assert runtime_config.out_dir is not None
    os.makedirs(runtime_config.out_dir, exist_ok=True)

    if not args.skip_default_style:
        default_style.save_styles_by_dirs(
            str(paths.wavs_dir),
            runtime_config.out_dir,
            config_path=str(paths.config_path),
            config_output_path=os.path.join(runtime_config.out_dir, "config.json"),
        )

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(local_rank)

    global global_step, resume_skip_batches
    writer = None
    writer_eval = None
    if rank == 0 and not args.speedup:
        # logger = get_logger(runtime_config.model_dir)
        # logger.info(hps)
        check_git_hash(model_dir)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(
        hps.data.training_files,
        hps.data,
        wavs_dir=paths.wavs_dir,
        spec_cache=runtime_config.spec_cache,
    )
    collate_fn = TextAudioSpeakerCollate()
    if not args.not_use_custom_batch_sampler:
        # バケット境界（スペクトログラムのフレーム数単位）
        # 44100Hz / hop_length=512 の場合: 1 frame ≈ 0.0116 秒
        # 上限 2153 frames ≈ 25.00 秒（25 秒の音声 = 1102500 サンプルは 1102500 // 512 = 2153 frames なので、25 秒ちょうどまでカバーできる）
        # 1000 frames 以降は長尺音声の絶対数が少ないため 200 frames 刻み (≈2.3 秒) に拡大
        bucket_boundaries = [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2153]  # fmt: skip
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            bucket_boundaries,
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            # batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
    else:
        train_sampler = DistributedLengthGroupedSampler(
            dataset=train_dataset,
            batch_size=hps.train.batch_size,
            num_replicas=n_gpus,
            rank=rank,
            lengths=train_dataset.lengths,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            # shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
        logger.info("Using DistributedLengthGroupedSampler for training.")
        logger.debug(f"len(train_dataset): {len(train_dataset)}")
        logger.debug(f"len(train_loader): {len(train_loader)}")

    eval_dataset = None
    eval_loader = None
    if rank == 0 and not args.speedup:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files,
            hps.data,
            wavs_dir=paths.wavs_dir,
            spec_cache=runtime_config.spec_cache,
        )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if hps.model.use_duration_discriminator is True:
        logger.info("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None
    if hps.model.use_spk_conditioned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        logger.info("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # hps.model 以下のすべての値を引数に渡す
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    ).cuda(local_rank)

    if getattr(hps.train, "freeze_ZH_bert", False):
        logger.info("Freezing ZH bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_EN_bert", False):
        logger.info("Freezing EN bert encoder !!!")
        for param in net_g.enc_p.en_bert_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("Freezing JP bert encoder !!!")
        for param in net_g.enc_p.ja_bert_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_style", False):
        logger.info("Freezing style encoder !!!")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_decoder", False):
        logger.info("Freezing decoder !!!")
        for param in net_g.dec.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    net_g = DDP(net_g, device_ids=[local_rank])
    net_d = DDP(net_d, device_ids=[local_rank])
    if net_dur_disc is not None:
        # NOTE: gin_channels != 0 (マルチスピーカー) 時、self.cond 層が作成されるが
        # forward 時に g を渡していないため未使用パラメータとなる
        # DDP でエラーを回避するため find_unused_parameters=True が必要
        net_dur_disc = DDP(
            net_dur_disc, device_ids=[local_rank], find_unused_parameters=True
        )

    resume_marker: dict[str, Any] | None = None
    if is_resuming(model_dir):
        # 再開用のチェックポイント一式 (G・D・DUR) を、揃っていて読める最新のステップから読み込む
        ## 書き込み途中で止まった版は飛ばして1つ前へ戻り、どれも読めなければ例外で止める (事前学習モデルすら読まないゼロからの学習へ黙って落とさない)
        resume_components: list[
            tuple[str, torch.nn.Module, torch.optim.Optimizer | None]
        ] = [
            ("G", net_g, optim_g),
            ("D", net_d, optim_d),
        ]
        if net_dur_disc is not None:
            resume_components.append(("DUR", net_dur_disc, optim_dur_disc))
        resume_step, resume_epoch, resume_learning_rates, resume_marker = (
            utils.checkpoints.load_resume_state(
                model_dir,
                resume_components,
                skip_optimizer=hps.train.skip_optimizer,
            )
        )
        # スケジューラーが初期学習率を参照するので、読み込んだ学習率を補う
        for prefix, _, optimizer in resume_components:
            if optimizer is not None and not optimizer.param_groups[0].get(
                "initial_lr"
            ):
                optimizer.param_groups[0]["initial_lr"] = resume_learning_rates[prefix]
        if resume_marker is not None:
            # 再開の印があれば、保存した時点の次のステップ・エポック・エポック内で済んだバッチ数からそのまま続ける
            epoch_str = int(resume_marker["epoch"])
            global_step = int(resume_marker["next_global_step"])
            resume_skip_batches = int(resume_marker["batches_done"])
        else:
            # 印の無い以前の版のチェックポイントは、ファイル名のステップまで学習済みとみなして次のステップから続ける
            epoch_str = max(resume_epoch, 1)
            global_step = resume_step + 1
            resume_skip_batches = global_step - (epoch_str - 1) * len(train_loader)
            if resume_skip_batches < 0:
                resume_skip_batches = 0
        # エポックを全て終えた時点の一式なら、次のエポックの頭から始める
        if resume_skip_batches >= len(train_loader):
            epoch_str += 1
            resume_skip_batches = 0
        logger.info(
            f"******************Found the model. Current epoch is {epoch_str}, global step is {global_step}, skipping {resume_skip_batches} batches already trained in this epoch*********************"
        )
    else:
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(pretrained_model_dir, "G_0.safetensors"), net_g
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(pretrained_model_dir, "D_0.safetensors"), net_d
            )
            if net_dur_disc is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(pretrained_model_dir, "DUR_0.safetensors"),
                    net_dur_disc,
                )
            logger.info("Loaded the pretrained models.")
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
        finally:
            epoch_str = 1
            global_step = 0

    def lr_lambda(epoch: int) -> float:
        """
        Learning rate scheduler for warmup and exponential decay.
        - During the warmup period, the learning rate increases linearly.
        - After the warmup period, the learning rate decreases exponentially.
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return float(hps.train.lr_decay) ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        assert optim_dur_disc is not None
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc,
            lr_lambda=lr_lambda,
            last_epoch=scheduler_last_epoch,
        )
    else:
        scheduler_dur_disc = None
    # NOTE: GradScaler は本来 FP16 (AMP) 用であり、BF16 では不要
    ## BF16 は FP32 と同じ動的レンジを持つため、勾配スケーリングは必要ない
    ## ただし、enabled=False で初期化すれば害はないため、現状維持としている
    scaler = GradScaler(device="cuda", enabled=not hps.train.bf16_run)
    logger.info("Start training.")

    diff = abs(
        epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader)
    )
    pbar: tqdm[Any] | None = None
    if not args.no_progress_bar:
        pbar = tqdm(
            total=global_step + diff,
            initial=global_step,
            smoothing=0.05,
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        )
    initial_step = global_step

    writers = (
        (writer, writer_eval)
        if writer is not None and writer_eval is not None
        else None
    )
    # 保存した時点の乱数の状態へ戻し、中断しなかった場合と同じ乱数で続きを学習する
    if resume_marker is not None and "rng" in resume_marker:
        torch.set_rng_state(resume_marker["rng"]["torch"])
        torch.cuda.set_rng_state_all(resume_marker["rng"]["cuda"])
        logger.info(
            "Restored the random number generator states saved with the checkpoint."
        )
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                runtime_config,
                (net_g, net_d, net_dur_disc),
                (optim_g, optim_d, optim_dur_disc),
                (scheduler_g, scheduler_d, scheduler_dur_disc),
                scaler,
                (train_loader, eval_loader),
                logger,
                writers,
                pbar,
                initial_step,
            )
        else:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                runtime_config,
                (net_g, net_d, net_dur_disc),
                (optim_g, optim_d, optim_dur_disc),
                (scheduler_g, scheduler_d, scheduler_dur_disc),
                scaler,
                (train_loader, None),
                None,
                None,
                pbar,
                initial_step,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            assert scheduler_dur_disc is not None
            scheduler_dur_disc.step()

        if epoch == hps.train.epochs:
            # Save the final models
            assert optim_g is not None
            utils.checkpoints.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"G_{global_step}.pth"),
            )
            assert optim_d is not None
            utils.checkpoints.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"D_{global_step}.pth"),
            )
            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                utils.checkpoints.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"DUR_{global_step}.pth"),
                )
            # 最後のエポックを終えた一式の印を書く (エポック内のバッチは全て済んでいる)
            utils.checkpoints.save_resume_marker(
                model_dir,
                global_step,
                {
                    "epoch": epoch,
                    "next_global_step": global_step,
                    "batches_done": len(train_loader),
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                    },
                },
            )
            utils.safetensors.save_safetensors(
                net_g,
                epoch,
                os.path.join(
                    runtime_config.out_dir,
                    f"{runtime_config.model_name}_e{epoch}_s{global_step}.safetensors",
                ),
                for_infer=True,
            )
            if runtime_config.repo_id is not None:
                future1 = api.upload_folder(
                    repo_id=runtime_config.repo_id,
                    folder_path=runtime_config.dataset_path,
                    path_in_repo=f"Data/{runtime_config.model_name}",
                    delete_patterns="*.pth",  # Only keep the latest checkpoint
                    ignore_patterns="raw/**",  # Ignore raw data
                    run_as_future=True,
                )
                future2 = api.upload_folder(
                    repo_id=runtime_config.repo_id,
                    folder_path=runtime_config.out_dir,
                    path_in_repo=f"model_assets/{runtime_config.model_name}",
                    run_as_future=True,
                )
                try:
                    future1.result()
                    future2.result()
                except Exception as ex:
                    logger.error("Failed to upload to HuggingFace", exc_info=ex)

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    rank: int,
    local_rank: int,
    epoch: int,
    hps: HyperParameters,
    runtime_config: TrainRuntimeConfig,
    nets: tuple[DDP, DDP, DDP | None],
    optims: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer | None,
    ],
    schedulers: tuple[
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.LambdaLR | None,
    ],
    scaler: GradScaler,
    loaders: tuple[DataLoader[Any], DataLoader[Any] | None],
    logger: LoguruLogger | None,
    writers: tuple[SummaryWriter, SummaryWriter] | None,
    pbar: tqdm[Any] | None,
    initial_step: int,
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    writer: SummaryWriter | None = None
    writer_eval: SummaryWriter | None = None
    if writers is not None:
        writer, writer_eval = writers

    # マルチ GPU 学習は基本行わないため、DistributedBucketSampler でのシャッフルを固定して再現性を優先する
    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step, resume_skip_batches

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        ja_bert,
        en_bert,
        style_vec,
    ) in enumerate(train_loader):
        # エポックの途中から再開したときは、中断前に学習済みの先頭のバッチを飛ばす (毎エポック同じ順に並ぶので、飛ばせば中断前と同じ順で続く)
        if batch_idx < resume_skip_batches:
            continue
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = (
            x.cuda(local_rank, non_blocking=True),
            x_lengths.cuda(local_rank, non_blocking=True),
        )
        spec, spec_lengths = (
            spec.cuda(local_rank, non_blocking=True),
            spec_lengths.cuda(local_rank, non_blocking=True),
        )
        y, y_lengths = (
            y.cuda(local_rank, non_blocking=True),
            y_lengths.cuda(local_rank, non_blocking=True),
        )
        speakers = speakers.cuda(local_rank, non_blocking=True)
        tone = tone.cuda(local_rank, non_blocking=True)
        language = language.cuda(local_rank, non_blocking=True)
        bert = bert.cuda(local_rank, non_blocking=True)
        ja_bert = ja_bert.cuda(local_rank, non_blocking=True)
        en_bert = en_bert.cuda(local_rank, non_blocking=True)
        style_vec = style_vec.cuda(local_rank, non_blocking=True)

        with autocast(
            device_type="cuda",
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16,
        ):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
                style_vec,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            loss_dur_disc_all: torch.Tensor | None = None
            loss_dur_gen: torch.Tensor | None = None

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(
                device_type="cuda",
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16,
            ):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw.detach(),
                    logw_.detach(),
                )
                with autocast(
                    device_type="cuda",
                    enabled=hps.train.bf16_run,
                    dtype=torch.bfloat16,
                ):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                assert optim_dur_disc is not None
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(
            device_type="cuda",
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16,
        ):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            y_dur_hat_g_gen: list[torch.Tensor] | None = None
            if net_dur_disc is not None:
                _, y_dur_hat_g_gen = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(
                device_type="cuda",
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16,
            ):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    assert y_dur_hat_g_gen is not None
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g_gen)
                    loss_gen_all += loss_dur_gen
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        # 勾配爆発を防ぐため、常に勾配クリッピングを適用
        # if getattr(hps.train, "bf16_run", False):
        torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0 and not runtime_config.speedup:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # logger.info(
                #     "Train Epoch: {} [{:.0f}%]".format(
                #         epoch, 100.0 * batch_idx / len(train_loader)
                #     )
                # )
                # logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                )
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                assert writer is not None
                summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )

            if (
                global_step % hps.train.eval_interval == 0
                and global_step != 0
                and initial_step != global_step
            ):
                if not runtime_config.speedup:
                    assert eval_loader is not None
                    assert writer_eval is not None
                    evaluate(hps, net_g, eval_loader, writer_eval)
                assert runtime_config.model_dir is not None
                utils.checkpoints.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(runtime_config.model_dir, f"G_{global_step}.pth"),
                )
                utils.checkpoints.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(runtime_config.model_dir, f"D_{global_step}.pth"),
                )
                if net_dur_disc is not None:
                    assert optim_dur_disc is not None
                    utils.checkpoints.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(
                            runtime_config.model_dir, f"DUR_{global_step}.pth"
                        ),
                    )
                # 一式を書き終えた印として、次に学習するステップ・エポック内で済んだバッチ数・乱数の状態を最後に保存する
                utils.checkpoints.save_resume_marker(
                    runtime_config.model_dir,
                    global_step,
                    {
                        "epoch": epoch,
                        "next_global_step": global_step + 1,
                        "batches_done": batch_idx + 1,
                        "rng": {
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all(),
                        },
                    },
                )
                if runtime_config.keep_ckpts > 0:
                    utils.checkpoints.clean_checkpoints(
                        model_dir_path=runtime_config.model_dir,
                        n_ckpts_to_keep=runtime_config.keep_ckpts,
                        sort_by_time=True,
                    )
                # Save safetensors (for inference) to `model_assets/{model_name}`
                utils.safetensors.save_safetensors(
                    net_g,
                    epoch,
                    os.path.join(
                        runtime_config.out_dir,
                        f"{runtime_config.model_name}_e{epoch}_s{global_step}.safetensors",
                    ),
                    for_infer=True,
                )
                if runtime_config.repo_id is not None:
                    api.upload_folder(
                        repo_id=runtime_config.repo_id,
                        folder_path=runtime_config.dataset_path,
                        path_in_repo=f"Data/{runtime_config.model_name}",
                        delete_patterns="*.pth",  # Only keep the latest checkpoint
                        ignore_patterns="raw/**",  # Ignore raw data
                        run_as_future=True,
                    )
                    api.upload_folder(
                        repo_id=runtime_config.repo_id,
                        folder_path=runtime_config.out_dir,
                        path_in_repo=f"model_assets/{runtime_config.model_name}",
                        run_as_future=True,
                    )

        global_step += 1
        if pbar is not None:
            pbar.set_description(
                f"Epoch {epoch}({100.0 * batch_idx / len(train_loader):.0f}%)/{hps.train.epochs}"
            )
            pbar.update()
    # 読み飛ばしは再開した最初のエポックだけで、次のエポックからは先頭から学習する
    resume_skip_batches = 0
    # 本家ではこれをスピードアップのために消すと書かれていたので、一応消してみる
    # と思ったけどメモリ使用量が減るかもしれないのでつけてみる
    gc.collect()
    torch.cuda.empty_cache()
    if pbar is None and rank == 0:
        assert logger is not None
        logger.info(f"====> Epoch: {epoch}, step: {global_step}")


def evaluate(
    hps: HyperParameters,
    generator: DDP,
    eval_loader: DataLoader[Any],
    writer_eval: SummaryWriter,
) -> None:
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print()
    logger.info("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
            en_bert,
            style_vec,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            en_bert = en_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    en_bert,
                    style_vec,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # mel = spec_to_mel_torch(
                #     spec,
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # y_hat_mel = mel_spectrogram_torch(
                #     y_hat.squeeze(1).float(),
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.hop_length,
                #     hps.data.win_length,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # image_dict.update(
                #     {
                #         f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             y_hat_mel[0].cpu().numpy()
                #         )
                #     }
                # )
                # image_dict.update(
                #     {
                #         f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             mel[0].cpu().numpy()
                #         )
                #     }
                # )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    run()
