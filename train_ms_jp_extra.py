"""
# 既知の問題点・バグ (対応済み)
---------------------------------------------------------------------------
## WavLM Discriminator の Generator 側損失が使用されない問題 (対応済み)
オリジナル Bert-VITS2 の JP-Extra 版から存在していたバグ。
WavLM Discriminator の損失（loss_lm, loss_lm_gen）を計算しても、Generator の学習に反映されなかった。
Duration Discriminator を意図的に無効化した際に、WavLM 損失の条件分岐が連動して無効になることが見落とされていたと推測される。
現在の事前学習モデルのベースである Bert-VITS2 時代からこの状態で学習されていたと思われる。

原因は、WavLM 損失を loss_gen_all に加算する処理が、
Duration Discriminator の条件分岐 `if net_dur_disc is not None:` の内部にネストされていたこと。
JP-Extra のデフォルト設定では `use_duration_discriminator: false` のため、
この条件分岐に入らず、WavLM 損失が計算されても使用されなかった。

これにより、WavLM Discriminator を有効にしていても Discriminator 側だけが鍛えられるだけで、
Generator は WavLM 損失を一切受け取れず、WavLM は学習に寄与しない状態だった。
にもかかわらず WavLM モデルのロード・forward pass・損失計算で数百 MB のメモリと計算時間を消費していた。
また TensorBoard には loss_lm, loss_lm_gen が記録されるが、実際には Generator の学習に使われていなかった。

### このコードでの対応
1. **ネストバグの修正**: WavLM 損失の加算を Duration Discriminator の条件分岐から独立させ、
   Duration Discriminator / WavLM Discriminator を個別に有効/無効化できるようにした。
2. **デフォルト設定の変更**: `use_wavlm_discriminator: false` に変更し、WavLM 関連のリソース消費を削減した。
   事前学習モデルの Generator は WavLM 損失を使用せずに学習されていたため、
   Generator の学習結果は従来と完全に等価である。

### WavLM を無効のままとする理由
事前学習モデルは WavLM なしで学習されたにもかかわらず、十分に高品質な音声を生成できている。
WavLM を有効化して学習に組み込むには JP-Extra ベースモデルの再学習が必要となる大変更になるため、
現時点では WavLM は存在しなかったものとして扱い、config.json で無効化している。
将来的に WavLM を導入する可能性は完全には排除しないが、積極的に予定はしていない。
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
import platform
from contextlib import nullcontext
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
from style_bert_vits2.models.models_jp_extra import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
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
from training.losses import (
    WavLMLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from training.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from training.runtime import (
    EMAModel,
    TrainRuntimeConfig,
)
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


# グローバル EMA インスタンス
ema_model: EMAModel | None = None


def run():
    paths_config = get_paths_config()
    parser = argparse.ArgumentParser(
        description="Train JP-Extra model.",
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
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average (EMA) for model weights. "
        "EMA weights are smoother and often produce more stable inference results.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate. Higher values (e.g., 0.9999) produce smoother weights. Default: 0.999",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights. "
        "Effective batch size = batch_size * gradient_accumulation_steps. Default: 1 (no accumulation)",
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
    collate_fn = TextAudioSpeakerCollate(use_jp_extra=True)
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
    if hps.model.use_wavlm_discriminator is True:
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        ).cuda(local_rank)
    else:
        net_wd = None
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

    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("Freezing (JP) bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
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
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None
    net_g = DDP(
        net_g,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    net_d = DDP(
        net_d,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            # bucket_cap_mb=512,
        )
    if net_wd is not None:
        net_wd = DDP(
            net_wd,
            device_ids=[local_rank],
            #  bucket_cap_mb=512
        )

    resume_step: int | None = None
    resume_marker: dict[str, Any] | None = None
    if is_resuming(model_dir):
        # 再開用のチェックポイント一式 (G・D・DUR・WD) を、揃っていて読める最新のステップから読み込む
        ## 書き込み途中で止まった版は飛ばして1つ前へ戻り、どれも読めなければ例外で止める (事前学習モデルすら読まないゼロからの学習へ黙って落とさない)
        resume_components: list[
            tuple[str, torch.nn.Module, torch.optim.Optimizer | None]
        ] = [
            ("G", net_g, optim_g),
            ("D", net_d, optim_d),
        ]
        if net_dur_disc is not None:
            resume_components.append(("DUR", net_dur_disc, optim_dur_disc))
        if net_wd is not None:
            resume_components.append(("WD", net_wd, optim_wd))
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
            if net_wd is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(pretrained_model_dir, "WD_0.safetensors"), net_wd
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
    if net_wd is not None:
        assert optim_wd is not None
        scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
            optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(local_rank)
    else:
        scheduler_wd = None
        wl = None

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

    # EMA (Exponential Moving Average) の初期化
    # 推論時により安定した出力を得るために、モデル重みの移動平均を保持
    global ema_model
    if args.use_ema:
        ema_model = EMAModel(net_g, decay=args.ema_decay)
        logger.info(
            f"[EMA] Enabled with decay: {args.ema_decay}. EMA weights will be saved for inference."
        )
    else:
        ema_model = None

    if ema_model is not None:
        try:
            # 再開したときは、G・D と同じステップの EMA を読む
            ema_path = (
                os.path.join(model_dir, f"EMA_{resume_step}.pth")
                if resume_step is not None
                else utils.checkpoints.get_latest_checkpoint_path(
                    model_dir, "EMA_*.pth"
                )
            )
            try:
                ema_state = torch.load(
                    ema_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                ema_state = torch.load(ema_path, map_location="cpu")
            ema_model.load_state_dict(ema_state.get("state_dict", {}))
            if "decay" in ema_state:
                ema_model.decay = float(ema_state["decay"])
            logger.info(f"[EMA] Loaded state from {ema_path}")
        except Exception as ex:
            logger.warning(ex)
            logger.warning("[EMA] Checkpoint not found. EMA will start fresh.")

    # 勾配累積のログ出力
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = hps.train.batch_size * args.gradient_accumulation_steps
        logger.info(
            f"[GradientAccumulation] Enabled with {args.gradient_accumulation_steps} steps. "
            f"Effective batch size: {hps.train.batch_size} x {args.gradient_accumulation_steps} = {effective_batch_size}"
        )

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
                (net_g, net_d, net_dur_disc, net_wd, wl),
                (optim_g, optim_d, optim_dur_disc, optim_wd),
                (scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd),
                scaler,
                (train_loader, eval_loader),
                logger,
                writers,
                pbar,
                initial_step,
                args.gradient_accumulation_steps,
            )
        else:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                runtime_config,
                (net_g, net_d, net_dur_disc, net_wd, wl),
                (optim_g, optim_d, optim_dur_disc, optim_wd),
                (scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd),
                scaler,
                (train_loader, None),
                None,
                None,
                pbar,
                initial_step,
                args.gradient_accumulation_steps,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            assert scheduler_dur_disc is not None
            scheduler_dur_disc.step()
        if net_wd is not None:
            assert scheduler_wd is not None
            scheduler_wd.step()

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
            if net_wd is not None:
                assert optim_wd is not None
                utils.checkpoints.save_checkpoint(
                    net_wd,
                    optim_wd,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"WD_{global_step}.pth"),
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
            # EMA が有効な場合は EMA 重みを保存
            model_to_save = (
                ema_model.get_ema_model() if ema_model is not None else net_g
            )
            utils.safetensors.save_safetensors(
                model_to_save,
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
    nets: tuple[DDP, DDP, DDP | None, DDP | None, Any | None],
    optims: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer | None,
        torch.optim.Optimizer | None,
    ],
    schedulers: tuple[
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.LambdaLR | None,
        torch.optim.lr_scheduler.LambdaLR | None,
    ],
    scaler: GradScaler,
    loaders: tuple[DataLoader[Any], DataLoader[Any] | None],
    logger: LoguruLogger | None,
    writers: tuple[SummaryWriter, SummaryWriter] | None,
    pbar: tqdm[Any] | None,
    initial_step: int,
    gradient_accumulation_steps: int = 1,
):
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
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
    if net_wd is not None:
        net_wd.train()

    # 勾配累積用の変数
    is_accumulating = gradient_accumulation_steps > 1

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
                (hidden_x, logw, logw_),  # , logw_sdp),
                g,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
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

        # 勾配累積を使用する場合は、累積ステップの最初でのみ zero_grad() を呼ぶ
        is_first_accumulation_step = (batch_idx % gradient_accumulation_steps) == 0
        is_last_accumulation_step = ((batch_idx + 1) % gradient_accumulation_steps) == 0
        should_step = is_last_accumulation_step or not is_accumulating

        loss_dur_disc_all: torch.Tensor | None = None
        losses_dur_disc_r: list[float] | None = None
        losses_dur_disc_g: list[float] | None = None
        loss_slm: torch.Tensor | None = None
        loss_lm: torch.Tensor | None = None
        loss_lm_gen: torch.Tensor | None = None
        loss_dur_gen: torch.Tensor | None = None
        losses_dur_gen: list[torch.Tensor] | None = None

        # Discriminator
        if is_first_accumulation_step:
            optim_d.zero_grad()
            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                optim_dur_disc.zero_grad()
            if net_wd is not None:
                assert optim_wd is not None
                optim_wd.zero_grad()
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
                logw_.detach(),
                logw.detach(),
                g.detach(),
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
        if net_wd is not None:
            # logger.debug(f"y.shape: {y.shape}, y_hat.shape: {y_hat.shape}")
            # shape: (batch, 1, time)
            with autocast(
                device_type="cuda",
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16,
            ):
                assert wl is not None
                loss_slm = wl.discriminator(
                    y.detach().squeeze(1), y_hat.detach().squeeze(1)
                ).mean()

        loss_disc_scaled = (
            loss_disc_all / gradient_accumulation_steps
            if is_accumulating
            else loss_disc_all
        )
        net_d_no_sync = net_d.no_sync if hasattr(net_d, "no_sync") else nullcontext
        with (
            net_d_no_sync()
            if is_accumulating and not is_last_accumulation_step
            else nullcontext()
        ):
            scaler.scale(loss_disc_scaled).backward()

        if net_dur_disc is not None:
            assert loss_dur_disc_all is not None
            loss_dur_disc_scaled = (
                loss_dur_disc_all / gradient_accumulation_steps
                if is_accumulating
                else loss_dur_disc_all
            )
            net_dur_no_sync = (
                net_dur_disc.no_sync
                if hasattr(net_dur_disc, "no_sync")
                else nullcontext
            )
            with (
                net_dur_no_sync()
                if is_accumulating and not is_last_accumulation_step
                else nullcontext()
            ):
                scaler.scale(loss_dur_disc_scaled).backward()

        if net_wd is not None:
            assert loss_slm is not None
            loss_slm_scaled = (
                loss_slm / gradient_accumulation_steps if is_accumulating else loss_slm
            )
            net_wd_no_sync = (
                net_wd.no_sync if hasattr(net_wd, "no_sync") else nullcontext
            )
            with (
                net_wd_no_sync()
                if is_accumulating and not is_last_accumulation_step
                else nullcontext()
            ):
                scaler.scale(loss_slm_scaled).backward()

        grad_norm_d = 0.0
        grad_norm_dur = 0.0
        grad_norm_wd = 0.0
        if should_step:
            scaler.unscale_(optim_d)
            if getattr(hps.train, "bf16_run", False):
                torch.nn.utils.clip_grad_norm_(
                    parameters=net_d.parameters(), max_norm=200
                )
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                scaler.unscale_(optim_dur_disc)
                # torch.nn.utils.clip_grad_norm_(
                # parameters=net_dur_disc.parameters(), max_norm=5
                # )
                grad_norm_dur = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

            if net_wd is not None:
                assert optim_wd is not None
                scaler.unscale_(optim_wd)
                grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
                scaler.step(optim_wd)

        with autocast(
            device_type="cuda",
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16,
        ):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            y_dur_hat_g_gen: list[torch.Tensor] | None = None
            if net_dur_disc is not None:
                _, y_dur_hat_g_gen = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
            if net_wd is not None:
                assert wl is not None
                loss_lm = wl(y.detach().squeeze(1), y_hat.squeeze(1)).mean()
                loss_lm_gen = wl.generator(y_hat.squeeze(1))
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
                # NOTE: loss_commit は JP-Extra 版では使用されておらず、config.json の "c_commit": 100 は参照されない
                ## オリジナル Bert-VITS2 では VQ (Vector Quantization) のコミットメント損失として
                ## 使用されていたが、JP-Extra 版では VQ が削除されたため不要になった
                # loss_commit = loss_commit * hps.train.c_commit

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

                # Duration Discriminator の損失を追加
                if net_dur_disc is not None:
                    assert y_dur_hat_g_gen is not None
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g_gen)
                    loss_gen_all += loss_dur_gen

                # ======================================================================
                ## オリジナルの Bert-VITS2 JP-Extra では、WavLM 損失 (loss_lm, loss_lm_gen) の
                ## 加算が Duration Discriminator の条件分岐の内部にネストされていた。
                ## 一方 JP-Extra 版のデフォルトハイパラは use_duration_discriminator: false のため、
                ## WavLM 損失が計算されても loss_gen_all には加算されない状態になっていた。
                ## このコードでは、WavLM 損失と Duration Discriminator 損失を独立して処理するため、
                ## Duration Discriminator / WavLM Discriminator を個別に有効/無効化できるようになっている。
                ## なお、事前学習モデルの Generator は WavLM 損失を使用せずに学習されてしまっていたと考えられるため、
                ## 現在のデフォルトハイパラでは use_wavlm_discriminator: false としている。そのため、この修正は実質的に影響しない。
                ## 将来的に WavLM 損失を有効化したモデルを学習する場合に、初めてこの修正が機能するようになる。
                # ======================================================================
                # WavLM Discriminator の損失を追加 (Duration Discriminator の有無に関係なく)
                if net_wd is not None:
                    assert loss_lm is not None
                    assert loss_lm_gen is not None
                    loss_gen_all += loss_lm + loss_lm_gen

        if is_first_accumulation_step:
            optim_g.zero_grad()

        # 勾配累積時は損失を累積ステップ数で割って平均化
        if is_accumulating:
            loss_gen_all = loss_gen_all / gradient_accumulation_steps

        net_g_no_sync = net_g.no_sync if hasattr(net_g, "no_sync") else nullcontext
        with (
            net_g_no_sync()
            if is_accumulating and not is_last_accumulation_step
            else nullcontext()
        ):
            scaler.scale(loss_gen_all).backward()

        # 累積ステップの最後でのみオプティマイザを更新
        grad_norm_g = 0.0
        if should_step:
            scaler.unscale_(optim_g)
            # 勾配爆発を防ぐため、常に勾配クリッピングを適用
            # if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            # EMA の更新（オプティマイザステップ後に実行）
            if ema_model is not None:
                ema_model.update(net_g)

        if rank == 0:
            if (
                should_step
                and global_step % hps.train.log_interval == 0
                and not runtime_config.speedup
            ):
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

                if net_dur_disc is not None:
                    assert loss_dur_disc_all is not None
                    assert losses_dur_disc_g is not None
                    assert losses_dur_disc_r is not None
                    assert loss_dur_gen is not None
                    assert losses_dur_gen is not None

                    scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})

                    scalar_dict.update(
                        {
                            f"loss/dur_disc_g/{i}": v
                            for i, v in enumerate(losses_dur_disc_g)
                        }
                    )
                    scalar_dict.update(
                        {
                            f"loss/dur_disc_r/{i}": v
                            for i, v in enumerate(losses_dur_disc_r)
                        }
                    )

                    scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
                    scalar_dict.update(
                        {f"loss/g/dur_gen_{i}": v for i, v in enumerate(losses_dur_gen)}
                    )

                # NOTE: 現在のコードでは、ここでログに記録される損失は実際に loss_gen_all に加算されて学習に使用されるようになっている
                ## 修正前は loss_lm, loss_lm_gen がログには表示されるが学習には使われない状態だった
                if net_wd is not None:
                    assert loss_slm is not None
                    assert loss_lm is not None
                    assert loss_lm_gen is not None
                    scalar_dict.update(
                        {
                            "loss/wd/total": loss_slm,
                            "grad_norm_wd": grad_norm_wd,
                            "loss/g/lm": loss_lm,
                            "loss/g/lm_gen": loss_lm_gen,
                        }
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
                if net_wd is not None:
                    assert optim_wd is not None
                    utils.checkpoints.save_checkpoint(
                        net_wd,
                        optim_wd,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(runtime_config.model_dir, f"WD_{global_step}.pth"),
                    )
                if ema_model is not None:
                    ema_state = {
                        "state_dict": ema_model.state_dict(),
                        "decay": ema_model.decay,
                    }
                    utils.checkpoints.save_torch_atomically(
                        ema_state,
                        os.path.join(
                            runtime_config.model_dir, f"EMA_{global_step}.pth"
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
                # EMA が有効な場合は EMA 重みを保存（より安定した推論結果が期待できる）
                model_to_save = (
                    ema_model.get_ema_model() if ema_model is not None else net_g
                )
                utils.safetensors.save_safetensors(
                    model_to_save,
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
            style_vec,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
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
