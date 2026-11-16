"""Rebase offset video clocks for the browser without transcoding."""

from .cli import open_local_video


def rebase_browser_video(source, destination, mime, first_timestamp):
    """Write a video-only playback copy with its first frame at zero."""
    import av

    formats = {"video/mp4": "mp4", "video/quicktime": "mp4", "video/webm": "webm"}
    format_name = formats.get(mime)
    if format_name is None:
        return None
    options = {"movflags": "faststart"} if format_name == "mp4" else {}
    with open_local_video(source) as original:
        video = original.streams.video[0]
        with av.open(str(destination), "w", format=format_name, options=options) as playback:
            output_video = playback.add_stream_from_template(video)
            for packet in original.demux(video):
                if packet.dts is None or packet.pts is None:
                    continue
                offset = round(first_timestamp / packet.time_base)
                packet.pts -= offset
                packet.dts -= offset
                packet.stream = output_video
                playback.mux(packet)
    return "video/mp4" if format_name == "mp4" else "video/webm"
