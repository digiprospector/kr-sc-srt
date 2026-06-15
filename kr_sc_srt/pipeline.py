from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import asr, download, media
from .jobs import job_name_from_source
from .segments import read_segments
from .srt import crop, read_srt, write_srt
from .state import JobState, StageResult, save_last_job
from .translate import translate_cues


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
        self.log = log
        self.state = JobState(self.out_dir / "run.json")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state.set_base(source=source, job_name=self.job_name, out_dir=str(self.out_dir))
        save_last_job(root, self.job_name, self.out_dir, source)

    def prepare(self, api_key: str | None, api_base: str, translation_model: str, asr_model: str) -> None:
        low_video = self._stage(
            "download_low",
            {"source": self.source, "quality": "low", "cookies": str(self.cookies) if self.cookies else None},
            [],
            lambda: StageResult({"video": str(download.resolve_source(self.source, self.out_dir, "low", self.cookies))}),
        )["video"]
        low_video_path = Path(low_video)

        audio_path = self.out_dir / "audio.wav"
        self._stage(
            "extract_audio",
            {"video": str(low_video_path)},
            [audio_path],
            lambda: StageResult({"audio": str(media.extract_audio(low_video_path, audio_path))}),
        )

        ko_srt = self.out_dir / "ko.srt"
        self._stage(
            "asr",
            {
                "audio": str(audio_path),
                "model": asr_model,
                "model_cache_dir": str(self.model_cache_dir) if self.model_cache_dir else None,
            },
            [ko_srt],
            lambda: StageResult(
                {
                    "srt": str(
                        asr.transcribe_to_srt(
                            audio_path,
                            ko_srt,
                            model_cache_dir=self.model_cache_dir,
                            model_name=asr_model,
                        )
                    )
                }
            ),
        )

        zh_srt = self.out_dir / "zh.srt"
        self._stage(
            "translate",
            {"source_srt": str(ko_srt), "api_base": api_base, "model": translation_model},
            [zh_srt],
            lambda: self._translate_stage(ko_srt, zh_srt, api_key, api_base, translation_model),
        )

    def render(self, segments_csv: Path, font: str) -> None:
        zh_srt = self.out_dir / "zh.srt"
        if not zh_srt.exists():
            raise FileNotFoundError(f"First pass subtitle not found: {zh_srt}")

        high_video = self._stage(
            "download_high",
            {"source": self.source, "quality": "high", "cookies": str(self.cookies) if self.cookies else None},
            [],
            lambda: StageResult({"video": str(download.resolve_source(self.source, self.out_dir, "high", self.cookies))}),
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

            self._stage(
                f"segment:{segment.safe_name}:srt",
                {"source_srt": str(zh_srt), "start_ms": segment.start_ms, "end_ms": segment.end_ms},
                [segment_srt],
                lambda segment=segment, segment_srt=segment_srt: self._segment_srt_stage(
                    source_cues, segment.start_ms, segment.end_ms, segment_srt
                ),
            )
            self._stage(
                f"segment:{segment.safe_name}:cut",
                {"video": str(high_video_path), "start_ms": segment.start_ms, "end_ms": segment.end_ms},
                [raw_clip],
                lambda segment=segment, raw_clip=raw_clip: StageResult(
                    {"video": str(media.cut_video(high_video_path, raw_clip, segment.start_ms, segment.end_ms))}
                ),
            )
            self._stage(
                f"segment:{segment.safe_name}:burn",
                {"video": str(raw_clip), "srt": str(segment_srt), "font": font},
                [final_video],
                lambda raw_clip=raw_clip, segment_srt=segment_srt, final_video=final_video: StageResult(
                    {"video": str(media.burn_subtitles(raw_clip, segment_srt, final_video, font=font))}
                ),
            )
            segment_results[segment.safe_name] = {"srt": str(segment_srt), "video": str(final_video)}

        self.state.data["segments"] = segment_results
        self.state.save()

    def _translate_stage(
        self,
        ko_srt: Path,
        zh_srt: Path,
        api_key: str | None,
        api_base: str,
        translation_model: str,
    ) -> StageResult:
        translated = translate_cues(read_srt(ko_srt), api_key=api_key, api_base=api_base, model=translation_model)
        write_srt(zh_srt, translated)
        return StageResult({"srt": str(zh_srt)})

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
            self.log(f"skip {name}")
            return dict(self.state.stage(name).get("outputs", {}))

        self.log(f"run {name}")
        self.state.mark_running(name, params)
        try:
            result = action()
            self.state.mark_completed(name, params, result)
            return result.outputs
        except BaseException as exc:
            self.state.mark_failed(name, params, exc)
            raise
