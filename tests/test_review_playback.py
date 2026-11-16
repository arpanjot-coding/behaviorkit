from io import BytesIO
from pathlib import Path

import pytest

from behaviorkit.video_setup import ReviewError, VideoReview
from test_pipeline import write_clip


def test_replacing_video_preserves_active_stream_until_closed(tmp_path):
    pytest.importorskip("cv2")
    clip = tmp_path / "local.mp4"
    write_clip(clip)
    data = clip.read_bytes()
    review = VideoReview(tmp_path / "models", tmp_path / "runs")
    try:
        staged = review.upload(BytesIO(data), len(data))
        assert staged["playback_available"]
        assert staged["playback_offset_seconds"] == 0
        with review.playback(staged["video_id"]) as (source, content_type):
            path = Path(source.name)
            assert content_type == "video/mp4"
            replacement = review.upload(BytesIO(data), len(data))
            assert replacement["video_id"] != staged["video_id"]
            assert path.is_file()
            with pytest.raises(ReviewError) as stale:
                with review.playback(staged["video_id"]):
                    pytest.fail("stale playback was accepted")
            assert stale.value.status == 404
            assert source.read(20) == data[:20]
        assert not path.exists()
    finally:
        review.close()
