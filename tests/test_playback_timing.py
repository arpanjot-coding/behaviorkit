from fractions import Fraction

import pytest

from behaviorkit.playback_timing import rebase_browser_video


def test_nonzero_pts_mp4_is_rebased_without_changing_decoded_frames(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    original = tmp_path / "offset.mp4"
    playback = tmp_path / "playback.mp4"
    with av.open(str(original), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        for index in range(10):
            rgb = np.full((96, 128, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = 50 + index, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    assert rebase_browser_video(original, playback, "video/mp4", 5) == "video/mp4"
    with av.open(str(original)) as original_video, av.open(str(playback)) as playback_video:
        before = list(original_video.decode(video=0))
        after = list(playback_video.decode(video=0))
        assert len(before) == len(after) == 10
        for source, browser in zip(before, after):
            assert source.time - 5 == pytest.approx(browser.time)
            assert np.array_equal(source.to_ndarray(), browser.to_ndarray())
