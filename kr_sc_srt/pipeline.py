from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

from . import asr, download, media
from .jobs import job_name_from_source
from .segments import Segment, read_segments
from .srt import crop, read_srt, write_srt
from .state import JobState, StageResult, save_last_job


class Pipeline:
    def __init__(
        self,
        root: Path,
        source: str,
        out_dir: Path | None = None,
        model_cache_dir: Path | None = None,
        cookies: Path | None = None,
        force_all: bool = False,
        force_stage: set[str] | None = None,
        test: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.root = root
        self.source = source
        self.job_name = job_name_from_source(source)
        self.out_dir = out_dir or root / "outputs" / self.job_name
        self.model_cache_dir = model_cache_dir
        self.cookies = cookies
        self.force_all = force_all
        self.force_stage = force_stage or set()
        self.test = test
        self.log = log
        self.state = JobState(self.out_dir / "run.json")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state.set_base(source=source, job_name=self.job_name, out_dir=str(self.out_dir))
        save_last_job(root, self.job_name, self.out_dir, source)
        self.log(f"[{_timestamp()}] 任务: {self.job_name}")
        self.log(f"[{_timestamp()}] 输出目录: {self.out_dir}")

    def prepare(self, asr_model: str, asr_chunk_s: int = asr.DEFAULT_CHUNK_S) -> None:
        self.log(f"[{_timestamp()}] prepare 阶段已启动")
        low_video = self._stage(
            "download_low",
            {"source": self.source, "quality": "low", "cookies": str(self.cookies) if self.cookies else None},
            [],
            lambda: StageResult({"video": str(download.resolve_source(self.source, self.out_dir, "low", stem=f"{self.job_name}.low", cookies=self.cookies))}),
        )["video"]
        low_video_path = Path(low_video)

        audio_path = self.out_dir / f"{self.job_name}.aac"
        limit_s = 600 if self.test else None
        self._stage(
            "extract_audio",
            {"video": str(low_video_path), "limit_s": limit_s},
            [audio_path],
            lambda: StageResult({"audio": str(media.extract_audio(low_video_path, audio_path, limit_s=limit_s))}),
        )

        ko_srt = self.out_dir / f"{self.job_name}.ko.srt"
        asr_params = {
            "audio": str(audio_path),
            "model": asr_model,
            "model_cache_dir": str(self.model_cache_dir) if self.model_cache_dir else None,
            "asr_chunk_s": asr_chunk_s,
        }

        # 如果在 Colab 中处于 CPU 运行模式，则停止执行并提示用户切换到 GPU 运行时
        import os
        is_colab = "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()
        is_forced = self.force_all or "asr" in self.force_stage
        if is_colab and (is_forced or not self.state.is_complete("asr", asr_params, [ko_srt])):
            has_gpu = False
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                pass
            if not has_gpu:
                raise RuntimeError(
                    "\n"
                    "=========================================================================\n"
                    "🔴 未检测到 GPU（当前激活的是 CPU 运行时）\n"
                    "=========================================================================\n"
                    "音频提取已成功完成。为了快速运行 ASR（语音转文字）并避免内存不足（OOM）错误，\n"
                    "请将运行时切换为 GPU：\n"
                    "\n"
                    "  1. 在顶部菜单栏中，转到：代码执行程序 (Runtime) -> 更改运行时类型 (Change runtime type)\n"
                    "  2. 选择硬件加速器：T4 GPU\n"
                    "  3. 点击：保存\n"
                    "  4. 重新运行此单元格以使用 GPU 恢复 ASR 识别（已完成的步骤将被自动跳过）。\n"
                    "========================================================================="
                )

        self._stage(
            "asr",
            asr_params,
            [ko_srt],
            lambda: StageResult(
                {
                    "srt": str(
                        asr.transcribe_to_srt(
                            audio_path,
                            ko_srt,
                            model_cache_dir=self.model_cache_dir,
                            model_name=asr_model,
                            chunk_duration_s=asr_chunk_s,
                        )
                    )
                }
            ),
        )

        self.log(f"[{_timestamp()}] 韩语 SRT 已就绪: {ko_srt}")
        self.log(f"[{_timestamp()}] 请在本地进行翻译，并将中文硬字幕保存为: {self.out_dir / f'{self.job_name}.zh.srt'}")
        self.log(f"[{_timestamp()}] prepare 阶段已完成")

    def render(self, segments_csv: Path, font: str) -> None:
        self.log(f"[{_timestamp()}] render 阶段已启动")
        zh_srt = self.out_dir / f"{self.job_name}.zh.srt"
        if not zh_srt.exists():
            raise FileNotFoundError(
                f"未找到中文硬字幕文件: {zh_srt}。请在本地翻译韩语 SRT，并将结果保存为 {self.job_name}.zh.srt。"
            )

        high_video = self._stage(
            "download_high",
            {"source": self.source, "quality": "high", "cookies": str(self.cookies) if self.cookies else None},
            [],
            lambda: StageResult({"video": str(download.resolve_source(self.source, self.out_dir, "high", stem=f"{self.job_name}.high", cookies=self.cookies))}),
        )["video"]
        high_video_path = Path(high_video)

        source_cues = read_srt(zh_srt)
        segment_dir = self.out_dir / "segments"
        cache_dir = segment_dir / ".cache"
        segment_results: dict[str, dict[str, str]] = {}
        for segment in read_segments(segments_csv):
            prefix = segment_dir / segment.safe_name
            raw_clip = cache_dir / f"{segment.safe_name}.source.mp4"
            segment_srt = prefix.with_suffix(".zh.srt")
            final_video = prefix.with_suffix(".mp4")

            def make_srt_action(s: Segment = segment, ss: Path = segment_srt) -> StageResult:
                return self._segment_srt_stage(source_cues, s.start_ms, s.end_ms, ss)

            def make_cut_action(s: Segment = segment, rc: Path = raw_clip) -> StageResult:
                return StageResult({"video": str(media.cut_video(high_video_path, rc, s.start_ms, s.end_ms))})

            def make_burn_action(rc: Path = raw_clip, ss: Path = segment_srt, fv: Path = final_video) -> StageResult:
                return StageResult({"video": str(media.burn_subtitles(rc, ss, fv, font=font))})

            self._stage(
                f"segment:{segment.safe_name}:srt",
                {"source_srt": str(zh_srt), "start_ms": segment.start_ms, "end_ms": segment.end_ms},
                [segment_srt],
                make_srt_action,
            )
            self._stage(
                f"segment:{segment.safe_name}:cut",
                {"video": str(high_video_path), "start_ms": segment.start_ms, "end_ms": segment.end_ms},
                [raw_clip],
                make_cut_action,
            )
            self._stage(
                f"segment:{segment.safe_name}:burn",
                {"video": str(raw_clip), "srt": str(segment_srt), "font": font},
                [final_video],
                make_burn_action,
            )
            segment_results[segment.safe_name] = {"srt": str(segment_srt), "video": str(final_video)}

        self.state.data["segments"] = segment_results
        self.state.save()
        self.log(f"[{_timestamp()}] render 阶段已完成")

    def _segment_srt_stage(self, source_cues, start_ms: int, end_ms: int, target: Path) -> StageResult:
        write_srt(target, crop(source_cues, start_ms, end_ms))
        return StageResult({"srt": str(target)})

    def _stage(
        self,
        name: str,
        params: dict,
        required_outputs: list[Path],
        action: Callable[[], StageResult],
    ) -> dict[str, str]:
        forced = self.force_all or name in self.force_stage
        if not forced and self.state.is_complete(name, params, required_outputs):
            self.log(f"[{_timestamp()}] 跳过 {name}")
            return dict(self.state.stage(name).get("outputs", {}))

        started = perf_counter()
        self.log(f"[{_timestamp()}] 开始 {name}")
        self.state.mark_running(name, params)
        try:
            result = action()
            self.state.mark_completed(name, params, result)
            elapsed = perf_counter() - started
            outputs = ", ".join(result.outputs.values())
            self.log(f"[{_timestamp()}] 完成 {name}，耗时 {elapsed:.1f}秒 -> {outputs}")
            return result.outputs
        except BaseException as exc:
            self.state.mark_failed(name, params, exc)
            elapsed = perf_counter() - started
            self.log(f"[{_timestamp()}] 失败 {name}，已耗时 {elapsed:.1f}秒: {exc}")
            raise


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")
