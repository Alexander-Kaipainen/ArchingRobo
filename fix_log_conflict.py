import os

fix_snippet = """
# Fix /tmp/cdds.LOG conflict for multiple DDS clients
try:
    import unitree_sdk2py.core.channel as dds_channel
    trace_config = getattr(dds_channel, "ChannelConfigHasInterface", "")
    if "/tmp/cdds.LOG" in trace_config:
        trace_path = f"/tmp/cdds.{os.getpid()}.LOG"
        dds_channel.ChannelConfigHasInterface = trace_config.replace("/tmp/cdds.LOG", trace_path)
except Exception:
    pass
"""

for fname in ["real_squat.py", "real_lateral_raise.py"]:
    with open(fname, "r") as f:
        text = f.read()
    
    if "cdds.LOG" not in text:
        text = text.replace(
            "from unitree_sdk2py.core.channel import ChannelPublisher",
            fix_snippet + "\nfrom unitree_sdk2py.core.channel import ChannelPublisher"
        )
        with open(fname, "w") as f:
            f.write(text)
